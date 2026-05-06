import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import models
import noise_schedule
import utils
import random

LOG2 = math.log(2)


def _sample_categorical(categorical_probs, num_samples=1):
  assert categorical_probs.ndim == 3
  categorical_probs = categorical_probs.repeat(
    num_samples, 1, 1)
  gumbel_norm = (
    1e-10
    - (torch.rand_like(categorical_probs) + 1e-10).log())
  return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
  return x.view(
    * x.shape,
    * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
  loss: torch.FloatTensor
  nlls: torch.FloatTensor
  token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
  pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> Tensor:
    """Computes the Perplexity.

    Returns:
     Perplexity
    """
    return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    self.tokenizer = tokenizer
    self.vocab_size = self.tokenizer.vocab_size
    self.sampler = self.config.sampling.predictor
    self.gen_ppl_eval_model_name_or_path = self.config.eval.\
      gen_ppl_eval_model_name_or_path
    self.antithetic_sampling = self.config.training.antithetic_sampling
    self.importance_sampling = self.config.training.importance_sampling
    self.change_of_variables = self.config.training.change_of_variables
    if (not hasattr(self.tokenizer, 'mask_token')
        or self.tokenizer.mask_token is None):
      self.mask_index = self.vocab_size
      self.vocab_size += 1
    else:
      self.mask_index = self.tokenizer.mask_token_id
    self.parameterization = self.config.parameterization
    if self.config.backbone == 'dit':
      self.backbone = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.backbone == 'dimamba':
      if models.dimamba is None:
        raise ImportError(
          "DiMamba backbone requires causal_conv1d which failed to import. "
          "Please reinstall causal_conv1d with: pip install causal_conv1d --no-cache-dir")
      self.backbone = models.dimamba.DiMamba(
        self.config,
        vocab_size=self.vocab_size,
        pad_token_id=self.tokenizer.pad_token_id)
    elif self.config.backbone == 'ar':
      self.backbone = models.autoregressive.AR(
        self.config,
        vocab_size=self.vocab_size,
        mask_index=self.mask_index)
    #### It seems that the hf_dit backbone is the only one that use the pretrained diffusion model as backbone.
    elif self.config.backbone == 'hf_dit':
      self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True)
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.backbone}')

    self.T = self.config.T
    self.subs_masking = self.config.subs_masking

    self.softplus = torch.nn.Softplus()
    # metrics are automatically reset at end of epoch
    metrics = torchmetrics.MetricCollection({
      'nll': NLL(),
      'bpd': BPD(),
      'ppl': Perplexity(),
    })
    metrics.set_dtype(torch.float64)
    self.train_metrics = metrics.clone(prefix='train/')
    self.valid_metrics = metrics.clone(prefix='val/')
    self.test_metrics = metrics.clone(prefix='test/')

    # generative perplexity
    self.gen_ppl_metric = Perplexity()
    self.entropy_metric = torchmetrics.aggregation.MeanMetric()
    self.time_metric = torchmetrics.aggregation.MeanMetric()
    self.eval_model_tokenizer = transformers.AutoTokenizer.\
      from_pretrained(self.gen_ppl_eval_model_name_or_path)
    if self.eval_model_tokenizer.pad_token is None:
      self.eval_model_tokenizer.pad_token =\
          self.eval_model_tokenizer.eos_token
      self.eval_model_tokenizer.pad_token_id =\
          self.eval_model_tokenizer.eos_token_id

    self.noise = noise_schedule.get_noise(self.config,
                                          dtype=self.dtype)
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None
    
    self.lr = self.config.optim.lr
    self.sampling_eps = self.config.training.sampling_eps
    self.time_conditioning = self.config.time_conditioning
    self.neg_infinity = -1000000.0
    self.fast_forward_epochs = None
    self.fast_forward_batches = None
    self._validate_configuration()

  def _validate_configuration(self):
    assert not (self.change_of_variables
                and self.importance_sampling)
    if self.parameterization == 'sedd':
      assert not self.importance_sampling
      assert not self.change_of_variables
    if self.parameterization == 'd3pm':
      assert self.T > 0
    if self.T > 0:
      assert self.parameterization in {'d3pm', 'subs'}
    if self.subs_masking:
      assert self.parameterization == 'd3pm'

  def on_load_checkpoint(self, checkpoint):
    if self.ema:
      self.ema.load_state_dict(checkpoint['ema'])
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
    self.fast_forward_epochs = checkpoint['loops'][
      'fit_loop']['epoch_progress']['current']['completed']
    self.fast_forward_batches = checkpoint['loops'][
      'fit_loop']['epoch_loop.batch_progress'][
        'current']['completed']

  def on_save_checkpoint(self, checkpoint):
    if self.ema:
      checkpoint['ema'] = self.ema.state_dict()
    # Copied from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
    # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
    # behind, so we're using the optimizer's progress.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['total'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total'][
              'completed'] * self.trainer.accumulate_grad_batches
    checkpoint['loops']['fit_loop'][
      'epoch_loop.batch_progress']['current'][
        'completed'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['current'][
              'completed'] * self.trainer.accumulate_grad_batches
    # _batches_that_stepped tracks the number of global steps, not the number
    # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
    checkpoint['loops']['fit_loop'][
      'epoch_loop.state_dict'][
        '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
          'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']
    if 'sampler' not in checkpoint.keys():
      checkpoint['sampler'] = {}
    if hasattr(self.trainer.train_dataloader.sampler,
               'state_dict'):
      sampler_state_dict = self.trainer.\
        train_dataloader.sampler.state_dict()
      checkpoint['sampler'][
        'random_state'] = sampler_state_dict.get(
          'random_state', None)
    else:
      checkpoint['sampler']['random_state'] = None

  def on_train_start(self):
    if self.ema:
      self.ema.move_shadow_params_to_device(self.device)
    # Adapted from:
    # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
    distributed = (
      self.trainer._accelerator_connector.use_distributed_sampler
      and self.trainer._accelerator_connector.is_distributed)
    if distributed:
      sampler_cls = dataloader.FaultTolerantDistributedSampler
    else:
      sampler_cls = dataloader.RandomFaultTolerantSampler
    updated_dls = []
    for dl in self.trainer.fit_loop._combined_loader.flattened:
      if hasattr(dl.sampler, 'shuffle'):
        dl_sampler = sampler_cls(
          dl.dataset, shuffle=dl.sampler.shuffle)
      else:
        dl_sampler = sampler_cls(dl.dataset)
      if (distributed
          and self.fast_forward_epochs is not None
          and self.fast_forward_batches is not None):
        dl_sampler.load_state_dict({
          'epoch': self.fast_forward_epochs,
          'counter': (self.fast_forward_batches
                      * self.config.loader.batch_size)})
      updated_dls.append(
        torch.utils.data.DataLoader(
          dl.dataset,
          batch_size=self.config.loader.batch_size,
          num_workers=self.config.loader.num_workers,
          pin_memory=self.config.loader.pin_memory,
          sampler=dl_sampler,
          shuffle=False,
          persistent_workers=True))
    self.trainer.fit_loop._combined_loader.flattened = updated_dls

  def optimizer_step(self, *args, **kwargs):
    super().optimizer_step(*args, **kwargs)
    if self.ema:
      self.ema.update(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))

  def _subs_parameterization(self, logits, xt):
    # log prob at the mask index = - infinity
    logits[:, :, self.mask_index] += self.neg_infinity
    
    # Normalize the logits such that x.exp() is
    # a probability distribution over vocab_size.
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)

    # Apply updates directly in the logits matrix.
    # For the logits of the unmasked tokens, set all values
    # to -infinity except for the indices corresponding to
    # the unmasked tokens.
    unmasked_indices = (xt != self.mask_index)
    logits[unmasked_indices] = self.neg_infinity
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

  def _d3pm_parameterization(self, logits):
    if self.subs_masking:
      logits[:, :, self.mask_index] += self.neg_infinity
    logits = logits - torch.logsumexp(logits, dim=-1,
                                      keepdim=True)
    return logits

  def _sedd_parameterization(self, logits, xt, sigma):
    esigm1_log = torch.where(
      sigma < 0.5,
      torch.expm1(sigma),
      sigma.exp() - 1).log().to(logits.dtype)
    # logits shape
    # (batch_size, diffusion_model_input_length, vocab_size)
    logits = logits - esigm1_log[:, None, None] - np.log(
      logits.shape[-1] - 1)
    # The below scatter operation sets the log score
    # for the input word to 0.
    logits = torch.scatter(logits, -1, xt[..., None],
                           torch.zeros_like(logits[..., :1]))
    return logits

  def _process_sigma(self, sigma):
    if sigma is None:
      assert self.parameterization == 'ar'
      return sigma
    if sigma.ndim > 1:
      sigma = sigma.squeeze(-1)
    if not self.time_conditioning:
      sigma = torch.zeros_like(sigma)
    assert sigma.ndim == 1, sigma.shape
    return sigma

  def forward(self, x, sigma):
    """Returns log score."""
    sigma = self._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
      logits = self.backbone(x, sigma)
    
    if self.parameterization == 'subs':
      return self._subs_parameterization(logits=logits,
                                         xt=x)
    elif self.parameterization == 'sedd':
      return self._sedd_parameterization(logits=logits,
                                         xt=x,
                                         sigma=sigma)
    elif self.parameterization == 'd3pm':
      return self._d3pm_parameterization(logits=logits)
    return logits

  def _d3pm_loss(self, model_output, xt, x0, t):
    dt = 1 / self.T

    if torch.is_tensor(t):
      t = t[:, None]
      assert t.ndim == 2
      t = t.clamp(0., 1. - 1e-4)
    alpha_t = 1 - t + torch.zeros_like(xt)
    alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

    log_x_theta_at_x0 = torch.gather(
      model_output, -1, x0[:, :, None]).squeeze(-1)
    log_x_theta_at_m = model_output[:, :, self.mask_index]
    x_theta_at_m = log_x_theta_at_m.exp()
    
    term_1_coef = dt / t
    term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
    term_1_log_dr = log_x_theta_at_x0
    
    term_2_coef = 1 - dt / t
    term_2_log_nr = term_1_log_nr
    term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

    L_vb_masked = (
      term_1_coef * (term_1_log_nr - term_1_log_dr)
      + term_2_coef * (term_2_log_nr - term_2_log_dr))

    L_vb = L_vb_masked * (xt == self.mask_index)

    return self.T * L_vb

  def _compute_loss(self, batch, prefix):
    if 'attention_mask' in batch:
      attention_mask = batch['attention_mask']
    else:
      attention_mask = None
    losses = self._loss(batch['input_ids'], attention_mask, prefix)
    loss = losses.loss

    if prefix == 'train':
      self.train_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.train_metrics
    elif prefix == 'val':
      self.valid_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.valid_metrics
    elif prefix == 'test':
      self.test_metrics.update(losses.nlls, losses.token_mask)
      metrics = self.test_metrics
    else:
      raise ValueError(f'Invalid prefix: {prefix}')

    self.log_dict(metrics,
                  on_step=False,
                  on_epoch=True,
                  sync_dist=True)
    return loss

  def on_train_epoch_start(self):
    self.backbone.train()
    self.noise.train()

  def training_step(self, batch, batch_idx):
    loss = self._compute_loss(batch, prefix='train')
    self.log(name='trainer/loss',
             value=loss.item(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return loss

  def on_validation_epoch_start(self):
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    assert self.valid_metrics.nll.mean_value == 0
    assert self.valid_metrics.nll.weight == 0

  def validation_step(self, batch, batch_idx):
    return self._compute_loss(batch, prefix='val')

  def on_validation_epoch_end(self):
    if ((self.config.eval.compute_perplexity_on_sanity
         or not self.trainer.sanity_checking)
         and self.config.eval.generate_samples
         and not self.parameterization == 'ar'):
      # TODO(justin): implement sampling and kv cache for AR
      samples, text_samples = None, None
      for _ in range(
        self.config.sampling.num_sample_batches):
        samples = self._sample()
        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)
        if self.config.eval.compute_generative_perplexity:
          self.compute_generative_perplexity(text_samples)
      if self.trainer.global_rank == 0 and hasattr(
        self.trainer.logger, 'log_table'):
        # Log the last generated samples
        text_samples = text_samples[
          : self.config.sampling.num_sample_log]
        self.trainer.logger.log_table(
          key=f'samples@global_step{self.global_step}',
          columns=['Generated Samples'],
          data=[[s] for s in text_samples])
      if self.config.eval.compute_generative_perplexity:
        self.log('val/gen_ppl',
                 self.gen_ppl_metric,
                 on_epoch=True,
                 on_step=False,
                 sync_dist=True)
    if self.ema:
      self.ema.restore(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()))

  def configure_optimizers(self):
    # TODO(yair): Lightning currently giving this warning when using `fp16`:
    #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
    #  Not clear if this is a problem or not.
    #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
    optimizer = torch.optim.AdamW(
      itertools.chain(self.backbone.parameters(),
                      self.noise.parameters()),
      lr=self.config.optim.lr,
      betas=(self.config.optim.beta1,
             self.config.optim.beta2),
      eps=self.config.optim.eps,
      weight_decay=self.config.optim.weight_decay)

    scheduler = hydra.utils.instantiate(
      self.config.lr_scheduler, optimizer=optimizer)
    scheduler_dict = {
      'scheduler': scheduler,
      'interval': 'step',
      'monitor': 'val/loss',
      'name': 'trainer/lr',
    }
    return [optimizer], [scheduler_dict]

  @torch.no_grad()
  def eval_retokenize(self, text_samples, max_length):
    """Retokenizes samples for the eval model.
    
    Args:
        text_samples: List of sentences generated by the model.
    Returns:
        samples: Samples re-tokenized for the eval model
        attn_mask: Attention mask for the eval model
        eval_context_size: Size of the context for the eval model
    """
    if 'llama2' in self.gen_ppl_eval_model_name_or_path:
      tokenizer_kwargs = {
        'text_samples': text_samples,
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 4096
    else:
      tokenizer_kwargs = {
        'return_tensors': 'pt',
        'return_token_type_ids': False,
        'return_attention_mask': True,
        'truncation': True,
        'padding': True,
        'max_length': max_length,
      }
      eval_context_size = 1024
    samples = self.eval_model_tokenizer(
      text_samples, ** tokenizer_kwargs)
    attn_mask = samples['attention_mask']
    samples = samples['input_ids']
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      attn_mask = attn_mask.to(self.device)
      samples = samples.to(self.device)      
    return samples, attn_mask, eval_context_size
  
  def compute_entropy(self, samples):
    for sample in samples:
      token_counts = torch.bincount(sample)
      token_counts = token_counts[token_counts > 0]
      token_probs = token_counts.float() / token_counts.sum()
      entropy = -torch.sum(token_probs * torch.log2(token_probs))
      self.entropy_metric.update(entropy)

  @torch.no_grad()
  def compute_generative_perplexity(
    self,
    text_samples: typing.List[str],
    retokenize: bool = True,
    max_length: typing.Optional[int] = None) -> None:
    """Compute the generative perplexity of the model.

    Args:
        text_samples: List of sentences generated by the model.
    
    Returns:
        Perplexity of the generated text under a different
        pre-trained AR model (e.g., GPT2).
    """
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    eval_model = transformers.AutoModelForCausalLM.from_pretrained(
      self.gen_ppl_eval_model_name_or_path).eval()
    if max_length is None:
      max_length = self.config.model.length
    if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
      eval_model = eval_model.to(self.device)
    # Re-tokenize using eval model's tokenizer
    if retokenize:
      (samples, attn_mask,
       eval_context_size) = self.eval_retokenize(
         text_samples, max_length=max_length)
    else:
      samples = text_samples
      attn_mask = torch.ones(samples.shape).to(self.device)
      eval_context_size = samples.shape[-1]
    batch_size = min(
      self.config.eval.perplexity_batch_size,
      samples.shape[0])
    num_batches = samples.shape[0] // batch_size
    for i in range(num_batches):
      _samples = torch.split(
        samples[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      _attn_mask = torch.split(
        attn_mask[i * batch_size: (i + 1) * batch_size],
        eval_context_size,
        dim=-1)
      for (sample_chunk, attn_mask_chunk) in zip(
        _samples, _attn_mask):
        logits = eval_model(
          sample_chunk, attention_mask=attn_mask_chunk)[0]
        logits = logits.transpose(-1, -2)
        
        nlls = F.cross_entropy(logits[..., :-1],
                               sample_chunk[..., 1:],
                               reduction='none')
        first_eos = (sample_chunk == self.eval_model_tokenizer\
                     .eos_token_id).cumsum(-1) == 1
        token_mask = (
          sample_chunk
          != self.eval_model_tokenizer.eos_token_id)
        self.gen_ppl_metric.update(
          nlls, first_eos[..., 1:] + token_mask[..., 1:])

  def q_xt(self, x, move_chance):
    """Computes the noisy sample xt.

    Args:
      x: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input. 
      move_chance: float torch.Tensor with shape (batch_size, 1).
    """
    move_indices = torch.rand(
      * x.shape, device=x.device) < move_chance
    xt = torch.where(move_indices, self.mask_index, x)
    return xt

  def _sample_prior(self, *batch_dims):
    return self.mask_index * torch.ones(
      * batch_dims, dtype=torch.int64)

  def _ddpm_caching_update(self, x, t, dt, p_x0=None):
    assert self.config.noise.type == 'loglinear'
    sigma_t, _ = self.noise(t)
    if t.ndim > 1:
      t = t.squeeze(-1)
    assert t.ndim == 1
    move_chance_t = t[:, None, None]
    move_chance_s = (t - dt)[:, None, None]
    assert move_chance_t.ndim == 3, move_chance_t.shape
    if p_x0 is None:
      p_x0 = self.forward(x, sigma_t).exp()
    
    assert move_chance_t.ndim == p_x0.ndim
    q_xs = p_x0 * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)
    
    copy_flag = (x != self.mask_index).to(x.dtype)
    return p_x0, copy_flag * x + (1 - copy_flag) * _x

  def _ddpm_update(self, x, t, dt):
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(t - dt)
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    assert sigma_t.ndim == 1, sigma_t.shape
    assert sigma_s.ndim == 1, sigma_s.shape
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    unet_conditioning = sigma_t
    log_p_x0 = self.forward(x, unet_conditioning)
    assert move_chance_t.ndim == log_p_x0.ndim
    # Technically, this isn't q_xs since there's a division
    # term that is missing. This division term doesn't affect
    # the samples.
    q_xs = log_p_x0.exp() * (move_chance_t
                             - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    _x = _sample_categorical(q_xs)

    copy_flag = (x != self.mask_index).to(x.dtype)
    return copy_flag * x + (1 - copy_flag) * _x

  def _ar_sampler(self, bsz):
    # precompute token buffer
    num_pred_tokens = self.config.model.length - 1
    x = torch.zeros(
      (bsz, num_pred_tokens + 1),
      dtype=torch.long,
      device=self.device)
    x[:, 0] = self.tokenizer.bos_token_id
    # precompute noise
    noise = (torch.distributions.Gumbel(0, 1)
             .sample((bsz, num_pred_tokens, self.vocab_size))
             .to(self.device))
    for i in range(num_pred_tokens):
      next_logits = self.forward(x[:, :i + 1], None)[:, -1]
      y = (next_logits + noise[:, i]).argmax(-1)
      x[:, i + 1] = y
    return x

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    if self.parameterization == 'ar':
      return self._ar_sampler(batch_size_per_gpu)
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm':
        x = self._ddpm_update(x, t, dt)
      elif self.sampler == 'ddpm_cache':
        p_x0_cache, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        x = self._analytic_update(x, t, dt)

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        x = self._denoiser_update(x, t)
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning).argmax(dim=-1)
    return x

  def restore_model_and_sample(self, num_steps, eps=1e-5):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    samples = self._sample(num_steps=num_steps, eps=eps)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return samples

  def get_score(self, x, sigma):
    model_output = self.forward(x, sigma)
    if self.parameterization == 'subs':
      # score(x, t) = p_t(y) / p_t(x)
      # => log score(x, t) = log p_t(y) - log p_t(x)
      
      # case 1: x = masked
      #   (i) y = unmasked
      #     log score(x, t) = log p_\theta(x)|_y + log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      #   (ii) y = masked
      #     log score(x, t) = 0

      # case 2: x = unmasked
      #   (i) y != masked, y != x
      #     log score(x_i, t) = - inf
      #   (ii) y = x 
      #     log score(x_i, t) = 0
      #   (iii) y = masked token
      #     log score(x_i, t) = - log k
      #     where k = exp(- sigma) / (1 - exp(- sigma))
      
      log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
      assert log_k.ndim == 1
      
      masked_score = model_output + log_k[:, None, None]
      masked_score[:, :, self.mask_index] = 0

      unmasked_score = self.neg_infinity * torch.ones_like(
        model_output)
      unmasked_score = torch.scatter(
        unmasked_score,
        -1,
        x[..., None],
        torch.zeros_like(unmasked_score[..., :1]))
      unmasked_score[:, :, self.mask_index] = - (
        log_k[:, None] * torch.ones_like(x))
      
      masked_indices = (x == self.mask_index).to(
        model_output.dtype)[:, :, None]
      model_output = (
        masked_score * masked_indices
        + unmasked_score * (1 - masked_indices))
    return model_output.exp()

  def _staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., self.mask_index] += extra_const
    return score

  def _analytic_update(self, x, t, step_size):
    curr_sigma, _ = self.noise(t)
    next_sigma, _ = self.noise(t - step_size)
    dsigma = curr_sigma - next_sigma
    score = self.get_score(x, curr_sigma)
    stag_score = self._staggered_score(score, dsigma)
    probs = stag_score * self._transp_transition(x, dsigma)
    return _sample_categorical(probs)

  def _denoiser_update(self, x, t):
    sigma, _ = self.noise(t)
    score = self.get_score(x, sigma)
    stag_score = self._staggered_score(score, sigma)
    probs = stag_score * self._transp_transition(x, sigma)
    probs[..., self.mask_index] = 0
    samples = _sample_categorical(probs)
    return samples

  def _transp_transition(self, i, sigma):
    sigma = _unsqueeze(sigma, reference=i[..., None])
    edge = torch.exp(-sigma) * F.one_hot(
      i, num_classes=self.vocab_size)
    edge += torch.where(i == self.mask_index,
                        1 - torch.exp(-sigma).squeeze(-1),
                        0)[..., None]
    return edge

  def _sample_t(self, n, device):
    _eps_t = torch.rand(n, device=device)
    if self.antithetic_sampling:
      offset = torch.arange(n, device=device) / n
      _eps_t = (_eps_t / n + offset) % 1
    t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
    if self.importance_sampling:
      return self.noise.importance_sampling_transformation(t)
    return t

  def _maybe_sub_sample(self, x0, attention_mask):
    seqlen = x0.shape[1]
    if seqlen > self.config.model.length:
      assert seqlen == 2 * self.config.model.length
      # cropping is needed for text8-crop dataset
      # try the same starting point for now
      start = np.random.choice(self.config.model.length)
      end = start + self.config.model.length
      input_tokens = x0[:, start: end]
      output_tokens = x0[:, start + 1: end + 1]
      new_attention_mask = attention_mask[:, start: end]

      # Helps with validation PPL, since the val
      # examples will all start and end with BOS/EOS
      input_tokens[:, 0] = self.tokenizer.bos_token_id
      output_tokens[:, -1] = self.tokenizer.eos_token_id
    elif self.parameterization == 'ar':
      input_tokens = x0[:, :-1]
      output_tokens = x0[:, 1:]
      new_attention_mask = attention_mask[:, 1:]
    else:
      input_tokens = x0
      output_tokens = None
      new_attention_mask = attention_mask
    return input_tokens, output_tokens, new_attention_mask

  def _reconstruction_loss(self, x0):
    t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                     device=self.device)
    assert self.config.noise.type == 'loglinear'
    # The above assert is for d3pm parameterization
    unet_conditioning = self.noise(t0)[0][:, None]
    model_output_t0 = self.forward(x0, unet_conditioning)
    return - torch.gather(input=model_output_t0,
                          dim=-1,
                          index=x0[:, :, None]).squeeze(-1)

  def _forward_pass_diffusion(self, x0, attention_mask=None, prefix=None):
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self.q_xt(x0, move_chance)
    model_output = self.forward(xt, unet_conditioning)
    utils.print_nans(model_output, 'model_output')

    if self.parameterization == 'sedd':
      return dsigma[:, None] * self._score_entropy(
        model_output, sigma[:, None], xt, x0)
    
    if self.T > 0:
      diffusion_loss = self._d3pm_loss(
        model_output=model_output, xt=xt, x0=x0, t=t)
      if self.parameterization == 'd3pm':
        reconstruction_loss = self._reconstruction_loss(x0)
      elif self.parameterization == 'subs':
        reconstruction_loss = 0
      return reconstruction_loss + diffusion_loss
    
    # SUBS parameterization, continuous time.
    log_p_theta = torch.gather(
      input=model_output,
      dim=-1,
      index=x0[:, :, None]).squeeze(-1)
    
    if self.change_of_variables or self.importance_sampling:
      return log_p_theta * torch.log1p(
        - torch.exp(- self.noise.sigma_min))
    
    return - log_p_theta * (
      dsigma / torch.expm1(sigma))[:, None]

  def _loss(self, x0, attention_mask, prefix=None):
    (input_tokens, output_tokens,
     attention_mask) = self._maybe_sub_sample(
       x0, attention_mask)

    if self.parameterization == 'ar':
      logprobs = self.backbone(input_tokens, None)
      loss = - logprobs.gather(
        -1, output_tokens[:, :, None])[:, :, 0]
    else:
      loss = self._forward_pass_diffusion(input_tokens, attention_mask, prefix)
    
    nlls = loss * attention_mask
    count = attention_mask.sum()

    batch_nll = nlls.sum()
    token_nll = batch_nll / count

    return Loss(loss=token_nll,
                nlls=nlls,
                token_mask=attention_mask)

  def _score_entropy(self, log_score, sigma, xt, x0):
    """Computes the SEDD loss.

    Args:
      log_score: float torch.Tensor with shape (batch_size,
          diffusion_model_input_length, vocab_size),
          log score, output of the denoising network.
      xt: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      x0: int torch.Tensor with shape (batch_size,
          diffusion_model_input_length), input.
      sigma: float torch.Tensor with shape (batch_size, 1).

    Returns:
      loss with shape (batch_size, diffusion_model_input_length)
    """
    masked_indices = xt == self.mask_index

    expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
    q_ratio = 1 / expsig_minus_1[masked_indices]

    words_that_were_masked = x0[masked_indices]

    neg_term = q_ratio * torch.gather(
      log_score[masked_indices],
      -1,
      words_that_were_masked[..., None]).squeeze(-1)
    score = log_score[masked_indices].exp()
    if self.mask_index == self.vocab_size - 1:
      pos_term = score[:, :-1].sum(dim=-1)
    else:
      pos_term = score[:, : self.mask_index].sum(
        dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
    const = q_ratio * (q_ratio.log() - 1)

    entropy = torch.zeros(* xt.shape, device=xt.device)
    entropy[masked_indices] += pos_term - neg_term + const
    return entropy

  @torch.no_grad
  def sample_subs_guidance(
    self, n_samples, stride_length, num_strides, dt=0.001):
    ones = torch.ones(n_samples, dtype=self.dtype,
                      device=self.device)

    num_steps = int(1 / dt)
    sampling_steps = 0
    intermediate_tokens = []
    target = None
    for _ in range(num_strides + 1):
      p_x0_cache = None
      x = self._sample_prior(
        n_samples,
        self.config.model.length).to(self.device)
      if target is not None:
        x[:, : -stride_length] = target
      for i in range(num_steps + 1):
        p_x0_cache, x_next = self._ddpm_caching_update(
          x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          p_x0_cache = None
          sampling_steps += 1
        x = x_next
      x = self.forward(x, 0 * ones).argmax(dim=-1)
      intermediate_tokens.append(
        x[:, :stride_length].cpu().numpy())
      target = x[:, stride_length:]
    
    intermediate_tokens.append(target.cpu().numpy())
    intermediate_text_samples = []
    sequence_lengths = ((
      np.concatenate(intermediate_tokens, axis=1)[:, 1:]
      == self.tokenizer.eos_token_id).cumsum(-1) == 0).sum(-1)
    for i in range(2, len(intermediate_tokens) + 1):
      intermediate_text_samples.append(
        self.tokenizer.batch_decode(
          np.concatenate(intermediate_tokens[:i], axis=1)))
    return (sampling_steps, intermediate_text_samples,
            sequence_lengths)

  def restore_model_and_semi_ar_sample(
      self, stride_length, num_strides, dt=0.001):
    """Generate samples from the model."""
    # Lightning auto-casting is not working in this method for some reason
    if self.ema:
      self.ema.store(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
      self.ema.copy_to(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.eval()
    self.noise.eval()
    (sampling_steps, samples,
     sequence_lengths) = self.sample_subs_guidance(
      n_samples=self.config.loader.eval_batch_size,
      stride_length=stride_length,
      num_strides=num_strides, 
      dt=dt)
    if self.ema:
      self.ema.restore(itertools.chain(
        self.backbone.parameters(),
        self.noise.parameters()))
    self.backbone.train()
    self.noise.train()
    return sampling_steps, samples, sequence_lengths


class EBM(Diffusion):

  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):

    import copy
    from omegaconf import open_dict

    ############################
    # Load pretrained diffusion as backbone
    ############################
    config_diffusion = copy.deepcopy(config)
    with open_dict(config_diffusion):
      config_diffusion.backbone = 'hf_dit' # Load pretrained diffusion as backbone
      config_diffusion.eval.checkpoint_path = 'kuleshov-group/mdlm-owt' # Path to the pretrained diffusion model
      # config_diffusion.eval.checkpoint_path = 's-sahoo/duo' # Path to the pretrained diffusion model
      # config_diffusion.eval.checkpoint_path = 'GSAI-ML/LLaDA-8B-Base' # Path to the pretrained diffusion model

    super().__init__(config_diffusion, tokenizer)

    self.backbone.eval()
    for p in self.backbone.parameters():
      p.requires_grad = False
    ############################
    # Finish loading pretrained diffusion as backbone
    ############################

    if self.config.ebm_backbone == 'hf_dit':
      self.ebm = transformers.AutoModelForMaskedLM.from_pretrained(
        config.eval.checkpoint_path, trust_remote_code=True).backbone
    elif self.config.ebm_backbone == 'dit':
      self.ebm = models.dit.DIT(
        self.config, vocab_size=self.vocab_size)
    elif self.config.ebm_backbone == 'ar':
      config_arebm = copy.deepcopy(self.config)
      with open_dict(config_arebm):
        config_arebm.model.causal = True
        config_arebm.backbone = 'ar'
      # Use configurable ar_checkpoint_path, fallback to default location
      ar_ckpt_path = getattr(config.eval, 'ar_checkpoint_path', 'Your AR Checkpoint Path')
      self.ebm = Diffusion.load_from_checkpoint(
        ar_ckpt_path,
        tokenizer=tokenizer,
        config=config_arebm).backbone
    else:
      raise ValueError(
        f'Unknown backbone: {self.config.ebm_backbone}')
    
    if self.config.ebm_backbone == 'dit' or self.config.ebm_backbone == 'hf_dit':
      self.ebm.vocab_proj = nn.Linear(
        2 * self.config.model.hidden_size, 
        self.config.model.hidden_size, 
        bias=True)
      from models.dit import DDitFinalLayer
      self.ebm.output_layer = DDitFinalLayer(
        self.config.model.hidden_size,
        self.config.model.hidden_size,
        self.config.model.cond_dim)
      self.ebm.energy_head = nn.Sequential(
        nn.Linear(config.model.hidden_size, config.model.hidden_size, bias=True),
        nn.ReLU(),
        nn.Linear(config.model.hidden_size, 1, bias=False),
      )

    self.backbone.ebm = self.ebm # Set the EBM as part of the backbone
    if self.config.training.ema > 0:
      self.ema = models.ema.ExponentialMovingAverage(
        itertools.chain(self.backbone.parameters(),
                        self.noise.parameters()),
        decay=self.config.training.ema)
    else:
      self.ema = None

  def _compute_ar_logits(self, context_tokens, output_tokens=None):
    """Compute AR model log probabilities given context tokens.
    
    Args:
      context_tokens: Input tokens for AR model (batch_size, seq_len)
      output_tokens: Optional output tokens for carry-over masking
    
    Returns:
      logits: Log probabilities (batch_size, seq_len, vocab_size)
    """
    # print(f"ar logits function input shape: context_tokens: {context_tokens.shape}, output_tokens: {output_tokens.shape}")
    # print(f"ar logits function input: context_tokens: {context_tokens}, output_tokens: {output_tokens}")
    # exit(0)
    x_emb = self.ebm.vocab_embed(context_tokens)
    rotary_cos_sin = self.ebm.rotary_emb(x_emb)
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      x = x_emb
      for i in range(len(self.ebm.blocks)):
        x = self.ebm.blocks[i](x, rotary_cos_sin, None, seqlens=None)
      output = self.ebm.output_layer(x, None)
    
    # log prob at the mask index = - infinity
    output[:, :, self.mask_index] = self.neg_infinity
    # Normalize to log probabilities
    logits = output - torch.logsumexp(output, dim=-1, keepdim=True)
    
    # Apply carry-over masking if enabled
    carry_over = self.config.sampling.ar_carry_over
    if carry_over and output_tokens is not None:
      unmasked_indices = (output_tokens != self.mask_index)
      logits[unmasked_indices] = self.neg_infinity
      logits[unmasked_indices, output_tokens[unmasked_indices]] = 0
    
    return logits

  def ebm_forward(self, xt, sigma, x0=None, log_p_x0=None, attention_mask=None):
    sigma = self._process_sigma(sigma)

    with torch.cuda.amp.autocast(dtype=torch.float32):
      indices = xt

      # rewrite the forward pass of the backbone
      if self.config.ebm_backbone == 'dit' or self.config.ebm_backbone == 'hf_dit':
        xt = self.ebm.vocab_embed(indices)
        x0 = self.ebm.vocab_embed(x0)
        x = self.ebm.vocab_proj(torch.cat([xt, x0], dim=-1))
        c = F.silu(self.ebm.sigma_map(sigma))

        rotary_cos_sin = self.ebm.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
          for i in range(len(self.ebm.blocks)):
            x = self.ebm.blocks[i](x, rotary_cos_sin, c, seqlens=None)
          x = self.ebm.output_layer(x, c)

          mean_pool = x.mean(dim=1)
          energy = self.ebm.energy_head(mean_pool)

      elif self.config.ebm_backbone == 'ar':
        parameterization = self.parameterization
        self.parameterization = 'ar'
        if attention_mask is None:
          attention_mask = torch.ones_like(x0)
        (x0_input_tokens, x0_output_tokens, _) = self._maybe_sub_sample(
          x0, attention_mask)
        (xt_input_tokens, xt_output_tokens, _) = self._maybe_sub_sample(
          xt, attention_mask)
        self.parameterization = parameterization

        x0_emb = self.ebm.vocab_embed(x0_input_tokens)
        x = x0_emb

        rotary_cos_sin = self.ebm.rotary_emb(x)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
          for i in range(len(self.ebm.blocks)):
            x = self.ebm.blocks[i](
              x, rotary_cos_sin, None, seqlens=None
            )
          output = self.ebm.output_layer(x, None)
        # log prob at the mask index = - infinity
        output[:, :, self.mask_index] = self.neg_infinity
        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = output - torch.logsumexp(output, dim=-1, keepdim=True)
        # Apply updates directly in the logits matrix.

        carry_over = self.config.sampling.ar_carry_over
        if carry_over:
          # For the logits of the unmasked tokens, set all values
          # to -infinity except for the indices corresponding to
          # the unmasked tokens.
          unmasked_indices = (xt_output_tokens != self.mask_index)
          logits[unmasked_indices] = self.neg_infinity
          logits[unmasked_indices, xt_output_tokens[unmasked_indices]] = 0

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
          energy_ar = (logits.gather(
            -1, x0_output_tokens[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)
          energy_diffusion = (log_p_x0.gather(
            -1, x0[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)
          energy = - energy_ar + energy_diffusion

      else:
        raise ValueError(
          f'Unknown backbone: {self.config.ebm_backbone}')
          
    return energy, energy_ar
  
  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples from the model."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    assert self.parameterization != 'ar'
    # Lightning auto-casting is not working in this method for some reason
    if num_steps is None:
      num_steps = self.config.sampling.steps
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    timesteps = torch.linspace(
      1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(
        x.shape[0], 1, device=self.device)
      if self.sampler == 'ddpm_cache':
        p_x0, x_next = self._ddpm_caching_update(
          x, t, dt, p_x0=p_x0_cache)
        if p_x0_cache is None:
          if t[0] > self.config.sampling.is_start or t[0] < self.config.sampling.is_end:
            p_x0_cache = p_x0
          else:
            # Energy-based Importance Sampling
            k = self.config.sampling.is_size
            x0_samples = _sample_categorical(
              p_x0, num_samples=k)  # (batch_size * k, seq_len)
            energy, ar_energy = self.ebm_forward(
              x.repeat(k, 1), t.repeat(k, 1), x0=x0_samples,
              log_p_x0=p_x0.repeat(k, 1, 1),
              attention_mask=torch.ones_like(x0_samples))
            energy = energy.view(x.shape[0], k)
            energy = energy - energy.max(dim=-1, keepdim=True)[0] # for numerical stability
            importance_weights = torch.softmax(
              energy / self.config.sampling.is_temp, dim=-1)
            x0_index = torch.multinomial(
              importance_weights, 1).view(x.shape[0])
            x0_samples = x0_samples.view(x.shape[0], k, -1)
            x0 = x0_samples[torch.arange(x.shape[0]), x0_index]
            p_x0_cache = F.one_hot(x0, num_classes=self.vocab_size).float()
            _, x_next = self._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
        if (not torch.allclose(x_next, x)
            or self.time_conditioning):
          # Disable caching
          p_x0_cache = None
        x = x_next
      else:
        raise ValueError(
          f'Unknown sampler: {self.sampler}')

    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                     device=self.device)
      if self.sampler == 'analytic':
        raise ValueError(
          f'Unknown sampler: {self.sampler}')
      else:
        unet_conditioning = self.noise(t)[0]
        x = self.forward(x, unet_conditioning).argmax(dim=-1)
    return x
  
  def _forward_pass_diffusion(self, x0, attention_mask=None, prefix=None):
    # Overwrite the forward pass of pure diffusion model

    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      # t \in {1/T, 2/T, ..., 1}
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    xt = self.q_xt(x0, move_chance)
    with torch.no_grad():
      log_p_x0 = self.forward(xt, unet_conditioning).detach()  # No update for diffusion model
    x0_pos = x0
    k = 1
    x0_neg = _sample_categorical(log_p_x0.exp(), num_samples=k)  # (batch_size * k, seq_len)
    energy_pos, pos_ar_energy = self.ebm_forward(xt, unet_conditioning, x0_pos, log_p_x0, attention_mask)
    energy_neg, neg_ar_energy = self.ebm_forward(xt.repeat(k, 1), 
                                  unet_conditioning.repeat(k, 1), 
                                  x0_neg, log_p_x0.repeat(k, 1, 1), 
                                  attention_mask.repeat(k, 1))
    energy_neg = energy_neg.view(x0.shape[0], k, -1)
    neg_ar_energy = neg_ar_energy.view(x0.shape[0], k, -1)
    
    energy_neg = energy_neg[:, 0]

    model_output = torch.cat([energy_pos, energy_neg], dim=0)
    utils.print_nans(model_output, 'model_output')

    assert self.parameterization == 'subs'

    if prefix == 'train':
      # Noise contrastive estimation
      loss = - (torch.log(torch.sigmoid(-energy_pos) + 1e-8) \
                + torch.log(torch.sigmoid(energy_neg) + 1e-8))
      
      assert loss.shape[-1] == 1 and loss.ndim == 2
      return loss
    elif prefix == 'val' or prefix == 'test':
      # NLL Estimation

      # Diffusion Term
      # SUBS parameterization, continuous time.
      if self.T == 0:
        log_p_theta = torch.gather(
          input=log_p_x0,
          dim=-1,
          index=x0[:, :, None]).squeeze(-1)
      elif self.T > 0:  # hard-coded for D3PM loss
        diffusion_loss = self._d3pm_loss(
          model_output=log_p_x0, xt=xt, x0=x0, t=t)
        # reweight for return call
        if self.change_of_variables or self.importance_sampling:
          log_p_theta = diffusion_loss / torch.log1p(
            - torch.exp(- self.noise.sigma_min))
        else:
          log_p_theta = - diffusion_loss / (
            dsigma / torch.expm1(sigma))[:, None]
      else:
        raise ValueError(
          f'Unknown T: {self.T}')

      # EBM Term
      if self.config.ebm_backbone in ['dit', 'hf_dit']:
        log_p_phi = - energy_pos + energy_neg
      elif self.config.ebm_backbone == 'ar':
        log_p_phi = - energy_pos  # self normalized, so ignore partition function
      # Assuming x0 is a full sequence of valid tokens
      log_p_phi = log_p_phi / log_p_theta.shape[-1]

      # if attention_mask is None:
      #   tmp_attention_mask = torch.ones_like(x0)
      # else:
      #   tmp_attention_mask = attention_mask
      # parameterization = self.parameterization
      # self.parameterization = 'ar'
      # (x0_input_tokens, x0_output_tokens, _) = self._maybe_sub_sample(x0, tmp_attention_mask)
      # (xt_input_tokens, xt_output_tokens, _) = self._maybe_sub_sample(xt, tmp_attention_mask)    
      # self.parameterization = parameterization

      # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      #   logits_ar_given_xt = self._compute_ar_logits(x0_input_tokens, xt_output_tokens)
      #   log_p_ar_xs_given_xt = (logits_ar_given_xt.gather(-1, x0_output_tokens[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)

      assert log_p_theta.ndim == log_p_phi.ndim
      log_p = log_p_theta + log_p_phi
      # log_p = pos_ar_energy / log_p_theta.shape[-1]
      # log_p = log_p_ar_xs_given_xt / log_p_theta.shape[-1]
      
      if self.change_of_variables or self.importance_sampling:
        return log_p * torch.log1p(
          - torch.exp(- self.noise.sigma_min))
      
      return - log_p * (
        dsigma / torch.expm1(sigma))[:, None]
    else:
      raise ValueError(
        f'Unknown prefix: {prefix}')


class InvariantEBM(EBM):
  """EBM with invariant energy function (Eq. 25).
  
  E(x0, xs, xt) = log(p_theta(xs|xt) / p_AR(xs|xt)) + log(p_theta(x0|xt) / p_AR(x0|xs)) - log F(xt)
  
  where:
    - xt: noisy tokens at time t
    - xs: intermediate tokens at time s (between t and 0)  
    - x0: clean tokens
    - p_theta: diffusion model
    - p_AR: autoregressive model
    - F(xt): partition function
  """

  def __init__(
    self,
    config,
    tokenizer: transformers.PreTrainedTokenizer):

    import copy
    from omegaconf import open_dict

    ############################
    # Load pretrained diffusion as backbone
    ############################
    config_diffusion = copy.deepcopy(config)
    with open_dict(config_diffusion):
      config_diffusion.backbone = 'hf_dit' # Load pretrained diffusion as backbone
      config_diffusion.eval.checkpoint_path = 'kuleshov-group/mdlm-owt' # Path to the pretrained diffusion model
      # config_diffusion.eval.checkpoint_path = 's-sahoo/duo' # Path to the pretrained diffusion model
      # config_diffusion.eval.checkpoint_path = 'GSAI-ML/LLaDA-8B-Base' # Path to the pretrained diffusion model

    super().__init__(config_diffusion, tokenizer)
    
    ###### no longer need to freeze the backbone because it is already frozen in the parent class
    # self.backbone.eval()
    # for p in self.backbone.parameters():
    #   p.requires_grad = False
    ############################
    # Finish loading pretrained diffusion as backbone
    ############################

    # Default intermediate time ratio (s = ratio * t)
    self.intermediate_time_ratio = getattr(
      config_diffusion.sampling, 'intermediate_time_ratio', -1)

  def _compute_ar_logits(self, context_tokens, output_tokens=None):
    """Compute AR model log probabilities given context tokens.
    
    Args:
      context_tokens: Input tokens for AR model (batch_size, seq_len)
      output_tokens: Optional output tokens for carry-over masking
    
    Returns:
      logits: Log probabilities (batch_size, seq_len, vocab_size)
    """
    x_emb = self.ebm.vocab_embed(context_tokens)
    rotary_cos_sin = self.ebm.rotary_emb(x_emb)
    
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      x = x_emb
      for i in range(len(self.ebm.blocks)):
        x = self.ebm.blocks[i](x, rotary_cos_sin, None, seqlens=None)
      output = self.ebm.output_layer(x, None)
    
    # log prob at the mask index = - infinity
    output[:, :, self.mask_index] = self.neg_infinity
    # Normalize to log probabilities
    logits = output - torch.logsumexp(output, dim=-1, keepdim=True)
    
    # Apply carry-over masking if enabled
    carry_over = self.config.sampling.ar_carry_over
    if carry_over and output_tokens is not None:
      unmasked_indices = (output_tokens != self.mask_index)
      logits[unmasked_indices] = self.neg_infinity
      logits[unmasked_indices, output_tokens[unmasked_indices]] = 0
    
    return logits

  def ebm_forward_invariant(
    self, 
    xt, 
    xs, 
    sigma_t, 
    sigma_s=None,
    x0=None, 
    log_p_x0_given_xt=None, 
    log_p_xs_given_xt=None, 
    attention_mask=None,
    return_terms=False,
    eval_mode=False):
    """Compute the invariant energy function (Eq. 25).
    
    E(x0, xs, xt) = log(p_theta(xs|xt) / p_AR(xs|xt)) + log(p_theta(x0|xt) / p_AR(x0|xs)) - log F(xt)
    
    Args:
      xt: Noisy tokens at time t (batch_size, seq_len)
      xs: Intermediate tokens at time s (batch_size, seq_len)
      sigma_t: Noise level at time t
      sigma_s: Noise level at time s (optional, for future use)
      x0: Clean tokens (batch_size, seq_len)
      log_p_x0_given_xt: Log p_theta(x0|xt) from diffusion model
      log_p_xs_given_xt: Log p_theta(xs|xt) from diffusion model
      attention_mask: Attention mask
      return_terms: If True, return (energy, term1, term2) instead of just energy
      eval_mode: If True, fix information leakage in term2 by:
        (a) disabling carry-over in the AR model for term2 (prevents the AR
            from trivially assigning prob=1 to positions unmasked in xs), and
        (b) restricting term2 to positions masked in xt (avoids counting
            positions the diffusion model predicts trivially).
        The analytical partition Z = exp(-term1) remains valid because the
        un-carried-over p_AR is still a proper normalised distribution.
    
    Returns:
      energy: The invariant energy (batch_size, 1)
      If return_terms=True, also returns term1 and term2 separately.
    """
    sigma_t = self._process_sigma(sigma_t)

    if attention_mask is None:
      attention_mask = torch.ones_like(x0)

    with torch.cuda.amp.autocast(dtype=torch.float32):
      if self.config.ebm_backbone != 'ar':
        raise ValueError(
          f'Invariant EBM only supports ar backbone, got: {self.config.ebm_backbone}')
      
      parameterization = self.parameterization
      self.parameterization = 'ar'
      
      (x0_input_tokens, x0_output_tokens, _) = self._maybe_sub_sample(x0, attention_mask)
      (xs_input_tokens, xs_output_tokens, _) = self._maybe_sub_sample(xs, attention_mask)
      (xt_input_tokens, xt_output_tokens, _) = self._maybe_sub_sample(xt, attention_mask)
      
      self.parameterization = parameterization

      with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # ============================================
        # Term 1: log(p_theta(xs|xt) / p_AR(xs|xt))
        # (unchanged between train and eval)
        # ============================================
        
        logits_ar_given_xt = self._compute_ar_logits(
          x0_input_tokens, xt_output_tokens)

        #########################################################
        ##### understand this one, based on the original code
        # logits_ar_given_xt = self._compute_ar_logits(x0_input_tokens, xt_output_tokens)
        #########################################################

        #########################################################
        #### if we think that the mask of xs is also need be predict use this:
        # log_p_ar_xs_given_xt = (logits_ar_given_xt.gather(
        #   -1, xs_output_tokens[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)
        #########################################################

        #########################################################
        #### if we think that the mask of xs is not need be predict use this:
        logits_ar_given_xt[:, :, self.mask_index] = self.neg_infinity
        ignore_indices = (xs_output_tokens == self.mask_index)
        logits_ar_given_xt[ignore_indices, xs_output_tokens[ignore_indices]] = 0

        log_p_ar_xs_given_xt = (logits_ar_given_xt.gather(
          -1, xs_output_tokens[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)

        log_p_theta_xs_given_xt = (log_p_xs_given_xt.gather(
          -1, xs[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)
        
        term1 = log_p_theta_xs_given_xt - log_p_ar_xs_given_xt
        
        # ============================================
        # Term 2: log(p_theta(x0|xt) / p_AR(x0|xs))
        # ============================================
        
        if eval_mode:
          # --- Evaluation mode: fair comparison ---
          # (a) No carry-over: the AR model must honestly predict all tokens,
          #     preventing the free pass at positions unmasked in xs.
          # To avoid data leakage, we use xt_output_tokens as carry-over tokens but we only predict the masked tokens in xs.
          logits_ar_given_xs = self._compute_ar_logits(
            x0_input_tokens, output_tokens=xt_output_tokens)
          
          # print(f"logits_ar_given_xs shape: {logits_ar_given_xs.shape}")
          # print(f"logits_ar_given_xs: {logits_ar_given_xs}")
          # print("--------------------------------")
          # (b) Restrict to positions masked in xt. Both sides of term2 are
          #     summed only over output positions where xt is masked, so the
          #     comparison is limited to positions the diffusion model truly
          #     needs to predict.  Unmasked-in-xt positions (where diffusion
          #     assigns prob ≈ 1) do not contribute and cannot inflate the
          #     correction.  The mask is based on xt_output_tokens (= xt[:,1:])
          #     so that diffusion and AR output positions are aligned.
          xt_out_mask = (xs_output_tokens == self.mask_index).float()
          
          log_p_ar_x0_per_tok = logits_ar_given_xs.gather(
            -1, x0_output_tokens[:, :, None])[:, :, 0]
          # print(f"log_p_ar_x0_per_tok shape: {log_p_ar_x0_per_tok.shape}")
          # print(f"log_p_ar_x0_per_tok: {log_p_ar_x0_per_tok}")
          # print("--------------------------------")
          log_p_ar_x0_given_xs = (
            log_p_ar_x0_per_tok * xt_out_mask
          ).sum(dim=-1, keepdim=True)
          
          log_p_theta_x0_per_tok = log_p_x0_given_xt.gather(
            -1, x0[:, :, None])[:, :, 0]
          log_p_theta_x0_given_xt = (
            log_p_theta_x0_per_tok[:, 1:] * xt_out_mask
          ).sum(dim=-1, keepdim=True)

          # log_p_theta_x0_given_xt = (log_p_x0_given_xt.gather(
          #   -1, x0[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)
        else:
          # --- Training mode: original behaviour ---
          logits_ar_given_xs = self._compute_ar_logits(
            xs_input_tokens, xs_output_tokens)

          log_p_ar_x0_given_xs = (logits_ar_given_xs.gather(
            -1, x0_output_tokens[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)
          
          log_p_theta_x0_given_xt = (log_p_x0_given_xt.gather(
            -1, x0[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)
        
        term2 = log_p_theta_x0_given_xt - log_p_ar_x0_given_xs
        
        energy = term1 + term2

    if return_terms:
      return energy, term1, term2, log_p_theta_x0_given_xt, log_p_ar_x0_given_xs
    return energy

  def _compute_log_p_xs_given_xt(self, log_p_x0_given_xt, xt, xs):
    """Compute log p(xs|xt), ignoring mask positions.
    
    For mask positions in xs, we set log prob = 0 so they don't contribute to the sum.
    This is equivalent to only computing probability over non-mask tokens.
    
    Args:
      log_p_x0_given_xt: Log probabilities from diffusion model (batch, seq, vocab)
      xt: Noisy tokens at time t (batch, seq)
      xs: Intermediate tokens at time s (batch, seq)
    
    Returns:
      log_p_xs_given_xt: Modified log probabilities (batch, seq, vocab)
    """
    # IMPORTANT: Clone to avoid in-place modification of the original tensor!
    log_p_xs_given_xt = log_p_x0_given_xt.clone()
    
    # Set mask token probability to -infinity (model shouldn't predict mask)
    log_p_xs_given_xt[:, :, self.mask_index] = self.neg_infinity
    
    # For positions where xs is mask, set probability to 0 (log prob = 0)
    # so they contribute 0 to the sum when gathered
    ignore_indices = (xs == self.mask_index)
    log_p_xs_given_xt[ignore_indices] = self.neg_infinity
    log_p_xs_given_xt[ignore_indices, xs[ignore_indices]] = 0
    
    return log_p_xs_given_xt

  @torch.no_grad()
  def _estimate_log_partition_function(
    self,
    xt,
    unet_conditioning,
    log_p_x0_given_xt,
    x0_fixed,
    xs_fixed,
    log_p_xs_fixed_given_xt,
    sigma_s,
    attention_mask,
    n_samples=16,
    use_leave_one_out=True):
    """Estimate log Z_phi(xs, xt) with xs fixed and x0_pos as an anchor.
    
    Includes x0_fixed (real data) as an additional sample alongside
    n_samples model-drawn x0's. This anchors the estimate with the
    most informative sample (real data typically has the lowest energy).
    
    Total samples used: n_samples (from model) + 1 (real x0) = n_total.
    
    By fixing xs, term1(xs, xt) is identical for all samples and cancels
    with energy_pos when computing -E_pos - log Z.
    
    Args:
      xt: Noisy tokens at time t (batch_size, seq_len)
      unet_conditioning: Noise conditioning for the model
      log_p_x0_given_xt: Log p_theta(x0|xt) from diffusion model (batch, seq, vocab)
      x0_fixed: Real x0 tokens to include as anchor sample (batch_size, seq_len)
      xs_fixed: Fixed intermediate tokens at time s (batch_size, seq_len)
      log_p_xs_fixed_given_xt: Pre-computed log p(xs|xt) for xs_fixed (batch, seq, vocab)
      sigma_s: Noise level at time s
      attention_mask: Attention mask (batch_size, seq_len)
      n_samples: Number of model-drawn x0 samples (total = n_samples + 1)
      use_leave_one_out: If True, use leave-one-out estimator for lower variance
    
    Returns:
      log_Z: Estimated log partition function (batch_size, 1)
    """
    batch_size = xt.shape[0]
    n_total = n_samples + 1  # model samples + 1 real x0
    
    # Sample n x0 candidates from p_theta(x0|xt)
    x0_model = _sample_categorical(
      log_p_x0_given_xt.exp(), num_samples=n_samples)  # (batch*n, seq)
    
    # Stack real x0 as the FIRST sample, then model samples
    # x0_fixed: (batch, seq) -> placed first
    # x0_model: (batch*n, seq) -> placed after
    x0_all = torch.cat([x0_fixed, x0_model], dim=0)  # (batch*(n+1), seq)
    
    # Compute energy for all (x0, xs_fixed, xt) tuples in one forward pass
    sample_energies = self.ebm_forward_invariant(
      xt=xt.repeat(n_samples, 1),
      xs=xs_fixed.repeat(n_samples, 1),
      sigma_t=unet_conditioning.repeat(n_samples, 1),
      sigma_s=sigma_s[:, None].repeat(n_samples, 1),
      x0=x0_model,
      log_p_x0_given_xt=log_p_x0_given_xt.repeat(n_samples, 1, 1),
      log_p_xs_given_xt=log_p_xs_fixed_given_xt.repeat(n_samples, 1, 1),
      attention_mask=attention_mask.repeat(n_samples, 1))
    
    pos_energy = self.ebm_forward_invariant(
      xt=xt,
      xs=xs_fixed,
      sigma_t=unet_conditioning,
      sigma_s=sigma_s[:, None],
      x0=x0_fixed,
      log_p_x0_given_xt=log_p_x0_given_xt,
      log_p_xs_given_xt=log_p_xs_fixed_given_xt,
      attention_mask=attention_mask)
    
    # Reshape to (batch_size, n_total)
    # Column 0 = real x0 energy, columns 1..n = model sample energies
    print(f"sample_energies shape: {sample_energies.shape}")
    print(f"sample_energies: {sample_energies}")
    print(f"pos_energy shape: {pos_energy.shape}")
    print(f"pos_energy: {pos_energy}")
    print("--------------------------------")
    neg_energies = -sample_energies.view(batch_size, n_samples)
    pos_energies = -pos_energy.view(batch_size, 1)
    energies = torch.cat([pos_energies, neg_energies], dim=-1)
    print(f"energies shape: {energies.shape}")
    print(f"energies: {energies}")
    print("--------------------------------")
    if use_leave_one_out and n_total > 2:
      # Leave-one-out estimator over all n_total samples
      log_Z_loo = torch.zeros(batch_size, n_total, device=xt.device)
      for i in range(n_total):
        mask = torch.ones(n_total, dtype=torch.bool, device=xt.device)
        mask[i] = False
        log_Z_loo[:, i] = (torch.logsumexp(energies[:, mask], dim=-1)
                           - math.log(n_total - 1))
      
      # Average over all leave-one-out estimates
      log_Z = log_Z_loo.mean(dim=-1, keepdim=True)  # (batch, 1)
    else:
      # Plug-in estimator: log Z = logsumexp(-E) - log(n_total)
      log_Z = (torch.logsumexp(energies, dim=-1, keepdim=True)
               - math.log(n_total))
    print(f"patition energy is: {torch.exp(log_Z)}")
    print("--------------------------------")
    return log_Z

  @torch.no_grad()
  def _estimate_log_partition_function_from_xs(
    self,
    xt,
    unet_conditioning,
    log_p_x0_given_xt,
    x0_fixed,
    xs_pos,
    move_chance_s,
    move_chance_t,
    sigma_s,
    attention_mask,
    n_samples=16,
    use_leave_one_out=True):
    """Estimate log Z by sampling different xs given fixed x0 and xt.
    
    Instead of varying x0, this varies xs while keeping x0 and xt fixed.
    Each xs sample respects the Markov chain x0 -> xs -> xt:
      - Positions unmasked in xt stay unmasked in xs (deterministic copy)
      - Positions masked in xt can be unmasked in xs (copied from x0)
        with probability proportional to (move_chance_t - move_chance_s)
      - Positions masked in xt can stay masked in xs
    
    Includes xs_pos (the original xs) as an anchor alongside n_samples
    newly sampled xs variants.
    
    Args:
      xt: Noisy tokens at time t (batch_size, seq_len)
      unet_conditioning: Noise conditioning for the model
      log_p_x0_given_xt: Log p_theta(x0|xt) (batch, seq, vocab)
      x0_fixed: Real x0 tokens (batch_size, seq_len)
      xs_pos: Original xs to include as anchor (batch_size, seq_len)
      move_chance_s: Masking probability at time s (batch, 1)
      move_chance_t: Masking probability at time t (batch, 1)
      sigma_s: Noise level at time s (batch,)
      attention_mask: Attention mask (batch_size, seq_len)
      n_samples: Number of additional xs samples (total = n_samples + 1)
      use_leave_one_out: If True, use leave-one-out estimator
    
    Returns:
      log_Z: Estimated log partition function (batch_size, 1)
    """
    batch_size = xt.shape[0]
    seq_len = xt.shape[1]
    n_total = n_samples + 1  # sampled xs + 1 original xs_pos
    
    # =========================================================
    # Sample n different xs that respect x0 -> xs -> xt
    #
    # The correct forward process is:
    #   1. Start from x0
    #   2. Mask each position independently with prob move_chance_s -> xs
    #   3. Mask additional positions with conditional prob -> xt
    #
    # Given fixed x0 and xt, we need xs consistent with both.
    # Constraint: unmasked in xt => unmasked in xs (copy from xt)
    # For positions masked in xt:
    #   P(xs=x0 | xt=mask) ∝ (move_chance_t - move_chance_s)
    #   P(xs=mask | xt=mask) ∝ move_chance_s
    # =========================================================
    
    is_masked_in_xt = (xt == self.mask_index)  # (batch, seq)
    
    # For masked positions in xt, compute probability that xs is unmasked
    # P(xs_j = x0_j | xt_j = mask) = (move_chance_t - move_chance_s) / move_chance_t
    # (because position was masked at time t, and the probability it was
    #  already unmasked at time s is the fraction of mask probability added
    #  between s and t)
    p_xs_unmasked_given_xt_masked = (
      (move_chance_t - move_chance_s) / (move_chance_t + 1e-8))  # (batch, 1)
    
    # Sample n_samples different xs by re-drawing the masking pattern
    # For each sample, independently decide which masked-in-xt positions
    # become unmasked in xs
    xs_samples_list = []
    for _ in range(n_samples):
      # Draw Bernoulli for each position: unmasked in xs?
      unmask_draw = (torch.rand(batch_size, seq_len, device=xt.device)
                     < p_xs_unmasked_given_xt_masked)
      
      # Build xs: start from xt (copies unmasked positions)
      xs_new = xt.clone()
      # For positions masked in xt AND drawn to be unmasked: copy from x0
      reveal_mask = is_masked_in_xt & unmask_draw
      xs_new[reveal_mask] = x0_fixed[reveal_mask]
      
      xs_samples_list.append(xs_new)
    
    # Stack: (batch * n_samples, seq)
    xs_sampled = torch.cat(xs_samples_list, dim=0)
    
    # Prepend xs_pos as anchor: total (batch * n_total, seq)
    xs_all = torch.cat([xs_pos, xs_sampled], dim=0)
    
    # Compute log_p_xs_given_xt for each xs variant
    log_p_xs_all = self._compute_log_p_xs_given_xt(
      log_p_x0_given_xt.repeat(n_total, 1, 1),
      xt.repeat(n_total, 1),
      xs_all)
    
    # Compute energy for all (x0_fixed, xs_i, xt) tuples
    energies = self.ebm_forward_invariant(
      xt=xt.repeat(n_total, 1),
      xs=xs_all,
      sigma_t=unet_conditioning.repeat(n_total, 1),
      sigma_s=sigma_s[:, None].repeat(n_total, 1),
      x0=x0_fixed.repeat(n_total, 1),
      log_p_x0_given_xt=log_p_x0_given_xt.repeat(n_total, 1, 1),
      log_p_xs_given_xt=log_p_xs_all,
      attention_mask=attention_mask.repeat(n_total, 1))
    
    # Reshape to (batch_size, n_total)
    # Column 0 = xs_pos energy, columns 1..n = sampled xs energies
    neg_energies = -energies.view(batch_size, n_total)
    
    # #region agent log
    import json as _json; _log_path = "Your Debug Log Path"
    _dbg = {"timestamp": 0, "location": "diffusion.py:_estimate_log_Z_from_xs", "runId": "run5", "hypothesisId": "xs_partition", "message": "partition_fn_from_xs", "data": {"neg_E_anchor_mean": float(neg_energies[:, 0].mean()), "neg_E_sampled_mean": float(neg_energies[:, 1:].mean()), "neg_E_std": float(neg_energies.std()), "batch_size": batch_size, "n_total": n_total}}
    with open(_log_path, "a") as _f: _f.write(_json.dumps(_dbg) + "\n")
    # #endregion

    if use_leave_one_out and n_total > 2:
      log_Z_loo = torch.zeros(batch_size, n_total, device=xt.device)
      for i in range(n_total):
        mask = torch.ones(n_total, dtype=torch.bool, device=xt.device)
        mask[i] = False
        log_Z_loo[:, i] = (torch.logsumexp(neg_energies[:, mask], dim=-1)
                           - math.log(n_total - 1))
      log_Z = log_Z_loo.mean(dim=-1, keepdim=True)
    else:
      log_Z = (torch.logsumexp(neg_energies, dim=-1, keepdim=True)
               - math.log(n_total))
    
    return log_Z

  @torch.no_grad()
  def _estimate_log_partition_function_from_definition(
    self,
    xt,
    log_p_x0_given_xt,
    x0_pos,
    xs_pos,
    term1_pos,
    attention_mask,
    n_samples=16,
    use_leave_one_out=True):
    """Estimate log Z from the paper's Equation (27).
    
    log Z = log E_{x0 ~ p(x0|xt)} exp(-E(x0, xs, xt))
    
    Since E = term1 + term2 and term1 is constant for fixed xs:
      log Z = -term1 + logsumexp(-term2_i) - log(n)
    where -term2_i = log p_theta(x0_i|xt) - log P_AR(x0_i|xs).
    
    Key advantage: The AR model only needs ONE forward pass on xs. Then we
    score n different x0 samples by gathering from the same logits.
    
    Includes x0_pos (real data) as an anchor sample.
    
    Args:
      xt: Noisy tokens at time t (batch_size, seq_len)
      log_p_x0_given_xt: Log p_theta(x0|xt) (batch, seq, vocab)
      x0_pos: Real x0 to include as anchor (batch_size, seq_len)
      xs_pos: Fixed intermediate tokens (batch_size, seq_len)
      term1_pos: Pre-computed term1 (batch, 1)
      attention_mask: Attention mask (batch_size, seq_len)
      n_samples: Number of model-drawn x0 samples (total = n_samples + 1 with anchor)
      use_leave_one_out: If True, use leave-one-out estimator
    
    Returns:
      log_Z: Estimated log partition function (batch_size, 1)
    """
    batch_size = xt.shape[0]
    n_total = n_samples + 1  # model samples + 1 real x0 anchor

    # =========================================================
    # Step 1: Run AR model ONCE on xs to get logits
    # =========================================================
    parameterization = self.parameterization
    self.parameterization = 'ar'
    (xs_input_tokens, xs_output_tokens, _) = self._maybe_sub_sample(
      xs_pos, attention_mask)
    self.parameterization = parameterization

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      logits_ar_given_xs = self._compute_ar_logits(
        xs_input_tokens, xs_output_tokens)
      # logits_ar_given_xs shape: (batch, seq_len-1, vocab)

    # =========================================================
    # Step 2: Sample n x0 from p_theta(x0|xt), stack with x0_pos
    # =========================================================
    x0_model = _sample_categorical(
      log_p_x0_given_xt.exp(), num_samples=n_samples)  # (batch*n, seq)
    x0_all = torch.cat([x0_pos, x0_model], dim=0)  # (batch*n_total, seq)

    # =========================================================
    # Step 3: For each x0, compute log P_AR(x0|xs) by gathering
    #         from the same AR logits (no extra forward pass!)
    # =========================================================
    self.parameterization = 'ar'
    (_, x0_all_output_tokens, _) = self._maybe_sub_sample(
      x0_all, attention_mask.repeat(n_total, 1))
    self.parameterization = parameterization

    logits_expanded = logits_ar_given_xs.repeat(n_total, 1, 1)
    log_p_ar_x0_given_xs = (logits_expanded.gather(
      -1, x0_all_output_tokens[:, :, None])[:, :, 0]
    ).sum(dim=-1, keepdim=True).view(batch_size, n_total)

    # =========================================================
    # Step 4: For each x0, compute log p_theta(x0|xt) by gathering
    # =========================================================
    log_p_xt_expanded = log_p_x0_given_xt.repeat(n_total, 1, 1)
    log_p_theta_x0_given_xt = (log_p_xt_expanded.gather(
      -1, x0_all[:, :, None])[:, :, 0]
    ).sum(dim=-1, keepdim=True).view(batch_size, n_total)

    # =========================================================
    # Step 5: Compute log Z using Eq. (27)
    #
    # Per-sample score = -term2_i = log p_theta(x0_i|xt) - log P_AR(x0_i|xs)
    # log Z = -term1 + logsumexp(-term2_i) - log(n)
    # =========================================================
    neg_term2 = log_p_theta_x0_given_xt - log_p_ar_x0_given_xs  # (batch, n_total)
    neg_term1 = -term1_pos  # (batch, 1)

    if use_leave_one_out and n_total > 2:
      log_Z_loo = torch.zeros(batch_size, n_total, device=xt.device)
      for i in range(n_total):
        mask = torch.ones(n_total, dtype=torch.bool, device=xt.device)
        mask[i] = False
        log_Z_loo[:, i] = (torch.logsumexp(neg_term2[:, mask], dim=-1)
                           - math.log(n_total - 1))
      log_Z_inner = log_Z_loo.mean(dim=-1, keepdim=True)
    else:
      log_Z_inner = (torch.logsumexp(neg_term2, dim=-1, keepdim=True)
                     - math.log(n_total))

    log_Z = neg_term1 + log_Z_inner
    return log_Z

  @torch.no_grad()
  def _estimate_log_Z_from_definition_diffusion_xs(
    self,
    xt,
    log_p_x0_given_xt,
    x0_pos,
    xs_pos,
    sigma_s,
    term1_pos,
    attention_mask,
    n_samples=16,
    use_leave_one_out=True):
    """Estimate log Z via Eq.(27), sampling x0 from the diffusion model
    conditioned on xs (not xt).
    
    Eq.(27): log Z = log E_{x0 ~ p(x0|xt)} exp(-E(x0, xs, xt))
    
    Proposal: q(x0) = p_theta(x0|xs), so we need importance weights:
      log Z = -term1 + logsumexp(log_w_i - term2_i) - log(n)
    where log_w_i = log p_theta(x0_i|xt) - log p_theta(x0_i|xs)
    and  -term2_i = log p_theta(x0_i|xt) - log P_AR(x0_i|xs)
    
    Combined per-sample score:
      log_w_i - term2_i = 2*log p_theta(x0_i|xt) - log p_theta(x0_i|xs) - log P_AR(x0_i|xs)
    
    Includes x0_pos (real data) as an anchor sample.
    
    Args:
      xt: Noisy tokens at time t (batch_size, seq_len)
      log_p_x0_given_xt: Log p_theta(x0|xt) (batch, seq, vocab)
      x0_pos: Real x0 to include as anchor (batch_size, seq_len)
      xs_pos: Fixed intermediate tokens (batch_size, seq_len)
      sigma_s: Noise level at time s (batch,) for conditioning diffusion on xs
      term1_pos: Pre-computed term1 (batch, 1)
      attention_mask: Attention mask (batch_size, seq_len)
      n_samples: Number of model-drawn x0 samples (total = n_samples + 1)
      use_leave_one_out: If True, use leave-one-out estimator
    
    Returns:
      log_Z: Estimated log partition function (batch_size, 1)
    """
    batch_size = xt.shape[0]
    n_total = n_samples + 1

    # =========================================================
    # Step 1: Get diffusion model prediction conditioned on xs
    # =========================================================
    sigma_s_cond = sigma_s[:, None] if sigma_s.ndim == 1 else sigma_s
    log_p_x0_given_xs = self.forward(xs_pos, sigma_s_cond).detach()
    # shape: (batch, seq, vocab)

    # =========================================================
    # Step 2: Sample x0 from p_theta(x0|xs), stack with x0_pos
    # =========================================================
    x0_model = _sample_categorical(
      log_p_x0_given_xs.exp(), num_samples=n_samples)  # (batch*n, seq)
    x0_all = torch.cat([x0_pos, x0_model], dim=0)  # (batch*n_total, seq)

    # =========================================================
    # Step 3: Run AR model ONCE on xs to get logits
    # =========================================================
    parameterization = self.parameterization
    self.parameterization = 'ar'
    (xs_input_tokens, xs_output_tokens, _) = self._maybe_sub_sample(
      xs_pos, attention_mask)
    self.parameterization = parameterization

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      logits_ar_given_xs = self._compute_ar_logits(
        xs_input_tokens, xs_output_tokens)

    # =========================================================
    # Step 4: Compute log P_AR(x0_i|xs) for each sample
    # =========================================================
    self.parameterization = 'ar'
    (_, x0_all_output_tokens, _) = self._maybe_sub_sample(
      x0_all, attention_mask.repeat(n_total, 1))
    self.parameterization = parameterization

    logits_expanded = logits_ar_given_xs.repeat(n_total, 1, 1)
    log_p_ar_x0_given_xs = (logits_expanded.gather(
      -1, x0_all_output_tokens[:, :, None])[:, :, 0]
    ).sum(dim=-1, keepdim=True).view(batch_size, n_total)

    # =========================================================
    # Step 5: Compute log p_theta(x0_i|xt) and log p_theta(x0_i|xs)
    # =========================================================
    log_p_xt_expanded = log_p_x0_given_xt.repeat(n_total, 1, 1)
    log_p_x0_given_xt_scores = (log_p_xt_expanded.gather(
      -1, x0_all[:, :, None])[:, :, 0]
    ).sum(dim=-1, keepdim=True).view(batch_size, n_total)

    log_p_xs_expanded = log_p_x0_given_xs.repeat(n_total, 1, 1)
    log_p_x0_given_xs_scores = (log_p_xs_expanded.gather(
      -1, x0_all[:, :, None])[:, :, 0]
    ).sum(dim=-1, keepdim=True).view(batch_size, n_total)

    # =========================================================
    # Step 6: Compute log Z using Eq. (27) with IS
    #
    # Per-sample score = log_w_i + (-term2_i)
    #   = (log p(x0|xt) - log p(x0|xs)) + (log p(x0|xt) - log P_AR(x0|xs))
    #   = 2*log p(x0|xt) - log p(x0|xs) - log P_AR(x0|xs)
    #
    # log Z = -term1 + logsumexp(score_i) - log(n)
    # =========================================================
    log_scores = (2 * log_p_x0_given_xt_scores
                  - log_p_x0_given_xs_scores
                  - log_p_ar_x0_given_xs)
    neg_term1 = -term1_pos  # (batch, 1)

    if use_leave_one_out and n_total > 2:
      log_Z_loo = torch.zeros(batch_size, n_total, device=xt.device)
      for i in range(n_total):
        mask = torch.ones(n_total, dtype=torch.bool, device=xt.device)
        mask[i] = False
        log_Z_loo[:, i] = (torch.logsumexp(log_scores[:, mask], dim=-1)
                           - math.log(n_total - 1))
      log_Z_inner = log_Z_loo.mean(dim=-1, keepdim=True)
    else:
      log_Z_inner = (torch.logsumexp(log_scores, dim=-1, keepdim=True)
                     - math.log(n_total))

    log_Z = neg_term1 + log_Z_inner
    return log_Z

  @torch.no_grad()
  def _estimate_log_Z_from_definition_ar_xs(
    self,
    xt,
    log_p_x0_given_xt,
    x0_pos,
    xs_pos,
    term1_pos,
    attention_mask,
    n_samples=16,
    use_leave_one_out=True):
    """Estimate log Z via Eq.(27), sampling x0 from the AR model
    conditioned on xs.
    
    Eq.(27): log Z = log E_{x0 ~ p(x0|xt)} exp(-E(x0, xs, xt))
    
    Proposal: q(x0) = p_AR(x0|xs), so we need importance weights:
      log Z = -term1 + logsumexp(log_w_i - term2_i) - log(n)
    where log_w_i = log p_theta(x0_i|xt) - log P_AR(x0_i|xs)
    and  -term2_i = log p_theta(x0_i|xt) - log P_AR(x0_i|xs)
    
    Note: log_w_i = -term2_i, so the combined per-sample score is:
      2 * (log p_theta(x0_i|xt) - log P_AR(x0_i|xs))
    
    Includes x0_pos (real data) as an anchor sample.
    
    Args:
      xt: Noisy tokens at time t (batch_size, seq_len)
      log_p_x0_given_xt: Log p_theta(x0|xt) (batch, seq, vocab)
      x0_pos: Real x0 to include as anchor (batch_size, seq_len)
      xs_pos: Fixed intermediate tokens (batch_size, seq_len)
      term1_pos: Pre-computed term1 (batch, 1)
      attention_mask: Attention mask (batch_size, seq_len)
      n_samples: Number of AR-drawn x0 samples (total = n_samples + 1)
      use_leave_one_out: If True, use leave-one-out estimator
    
    Returns:
      log_Z: Estimated log partition function (batch_size, 1)
    """
    batch_size = xt.shape[0]
    n_total = n_samples + 1

    # =========================================================
    # Step 1: Run AR model on xs to get logits for sampling
    # =========================================================
    parameterization = self.parameterization
    self.parameterization = 'ar'
    (xs_input_tokens, xs_output_tokens, _) = self._maybe_sub_sample(
      xs_pos, attention_mask)
    self.parameterization = parameterization

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      logits_ar_given_xs = self._compute_ar_logits(
        xs_input_tokens, xs_output_tokens)
      # shape: (batch, seq_len-1, vocab)

    # =========================================================
    # Step 2: Sample x0 from p_AR(x0|xs) using the AR logits
    # =========================================================
    ar_probs = logits_ar_given_xs.exp()  # (batch, seq_len-1, vocab)
    x0_ar_output = _sample_categorical(
      ar_probs, num_samples=n_samples)  # (batch*n, seq_len-1)

    first_token = x0_pos[:, :1].repeat(n_samples, 1)  # (batch*n, 1)
    x0_model = torch.cat([first_token, x0_ar_output], dim=1)  # (batch*n, seq)
    x0_all = torch.cat([x0_pos, x0_model], dim=0)  # (batch*n_total, seq)

    # =========================================================
    # Step 3: Score each x0 with log p_theta(x0|xt)
    # =========================================================
    log_p_xt_expanded = log_p_x0_given_xt.repeat(n_total, 1, 1)
    log_p_theta_x0_given_xt = (log_p_xt_expanded.gather(
      -1, x0_all[:, :, None])[:, :, 0]
    ).sum(dim=-1, keepdim=True).view(batch_size, n_total)

    # =========================================================
    # Step 4: Score each x0 with log P_AR(x0|xs) by gathering
    #         from the same AR logits
    # =========================================================
    self.parameterization = 'ar'
    (_, x0_all_output_tokens, _) = self._maybe_sub_sample(
      x0_all, attention_mask.repeat(n_total, 1))
    self.parameterization = parameterization

    logits_expanded = logits_ar_given_xs.repeat(n_total, 1, 1)
    log_p_ar_x0_given_xs = (logits_expanded.gather(
      -1, x0_all_output_tokens[:, :, None])[:, :, 0]
    ).sum(dim=-1, keepdim=True).view(batch_size, n_total)

    # =========================================================
    # Step 5: Compute log Z using Eq. (27) with IS
    #
    # Per-sample score = log_w_i + (-term2_i)
    #   = 2 * (log p_theta(x0|xt) - log P_AR(x0|xs))
    #
    # log Z = -term1 + logsumexp(score_i) - log(n)
    # =========================================================
    log_scores = 2 * (log_p_theta_x0_given_xt - log_p_ar_x0_given_xs)
    neg_term1 = -term1_pos  # (batch, 1)

    if use_leave_one_out and n_total > 2:
      log_Z_loo = torch.zeros(batch_size, n_total, device=xt.device)
      for i in range(n_total):
        mask = torch.ones(n_total, dtype=torch.bool, device=xt.device)
        mask[i] = False
        log_Z_loo[:, i] = (torch.logsumexp(log_scores[:, mask], dim=-1)
                           - math.log(n_total - 1))
      log_Z_inner = log_Z_loo.mean(dim=-1, keepdim=True)
    else:
      log_Z_inner = (torch.logsumexp(log_scores, dim=-1, keepdim=True)
                     - math.log(n_total))

    log_Z = neg_term1 + log_Z_inner
    return log_Z

  def _sample_xs_from_xt(self, xt, t, s, log_p_x0=None):
    """Sample intermediate state xs at time s given xt at time t.
    
    Uses DDPM transition kernel to go from time t to time s.
    Ensures proper relationship: unmasked tokens in xt stay unmasked in xs.
    
    Args:
      xt: Tokens at time t (batch_size, seq_len)
      t: Current time (batch_size, 1)
      s: Target intermediate time (batch_size, 1)
      log_p_x0: Optional cached log p(x0|xt)
    
    Returns:
      xs: Sampled tokens at time s
      log_p_xs_given_xt: Log transition probabilities p(xs|xt)
    """
    sigma_t, _ = self.noise(t)
    sigma_s, _ = self.noise(s)
    
    if sigma_t.ndim > 1:
      sigma_t = sigma_t.squeeze(-1)
    if sigma_s.ndim > 1:
      sigma_s = sigma_s.squeeze(-1)
    
    move_chance_t = 1 - torch.exp(-sigma_t)
    move_chance_s = 1 - torch.exp(-sigma_s)
    move_chance_t = move_chance_t[:, None, None]
    move_chance_s = move_chance_s[:, None, None]
    
    if log_p_x0 is None:
      unet_conditioning = sigma_t
      log_p_x0 = self.forward(xt, unet_conditioning)
    
    # Compute transition probabilities for going from t to s
    # For masked positions: P(xs=v|xt) ∝ p_theta(v|xt) * (t - s), P(xs=mask|xt) ∝ s
    q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
    q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
    
    # Sample xs for masked positions
    xs = _sample_categorical(q_xs)
    
    # Copy unmasked tokens from xt (deterministic transition)
    copy_flag = (xt != self.mask_index).to(xt.dtype)
    xs = copy_flag * xt + (1 - copy_flag) * xs
    
    # Compute log p(xs|xt) using the proper transition kernel
    log_p_xs_given_xt = self._compute_log_p_xs_given_xt(
      log_p_x0, xt, xs)
    
    return xs, log_p_xs_given_xt

  @torch.no_grad()
  def _sample(self, num_steps=None, eps=1e-5):
    """Generate samples using invariant energy-based importance sampling."""
    batch_size_per_gpu = self.config.loader.eval_batch_size
    assert self.parameterization != 'ar'
    
    if num_steps is None:
      num_steps = self.config.sampling.steps
    
    x = self._sample_prior(
      batch_size_per_gpu,
      self.config.model.length).to(self.device)
    
    timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
    dt = (1 - eps) / num_steps
    p_x0_cache = None

    for i in range(num_steps):
      t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
      
      if self.sampler == 'ddpm_cache':
        p_x0, x_next = self._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
        
        if p_x0_cache is None:
          # Check if we should apply invariant energy-based importance sampling
          is_start = self.config.sampling.is_start
          is_end = self.config.sampling.is_end
          
          if t[0] > is_start or t[0] < is_end:
            # Outside IS range, just cache p_x0
            p_x0_cache = p_x0
          else:
            # Apply Invariant Energy-based Importance Sampling
            k = self.config.sampling.is_size
            
            # Sample k candidates for x0
            x0_samples = _sample_categorical(p_x0, num_samples=k)  # (batch_size * k, seq_len)
            
            # Compute intermediate time s
            # s_ratio = self.intermediate_time_ratio
            if self.intermediate_time_ratio > 0:
              s_ratio = self.intermediate_time_ratio
            else:
              s_ratio = random.random()
              # s_ratio = random.uniform(0, 0.8)
            s = t * s_ratio  # s is between 0 and t
            
            # Sample xs for each x0 candidate
            # First, we need to get xs samples
            xs_samples, log_p_xs = self._sample_xs_from_xt(
              x.repeat(k, 1), 
              t.repeat(k, 1), 
              s.repeat(k, 1),
              log_p_x0=p_x0.repeat(k, 1, 1))
            
            # Compute invariant energy for each candidate
            energy = self.ebm_forward_invariant(
              xt=x.repeat(k, 1),
              xs=xs_samples,
              sigma_t=t.repeat(k, 1),
              sigma_s=s.repeat(k, 1),
              x0=x0_samples,
              log_p_x0_given_xt=p_x0.repeat(k, 1, 1),
              log_p_xs_given_xt=log_p_xs,
              attention_mask=torch.ones_like(x0_samples))
            
            # Reshape energy and compute importance weights
            # We want weights ∝ exp(-E) since lower energy = higher probability
            energy = energy.view(x.shape[0], k)
            neg_energy = -energy
            neg_energy = neg_energy - neg_energy.max(dim=-1, keepdim=True)[0]  # Numerical stability
            
            importance_weights = torch.softmax(
              neg_energy / self.config.sampling.is_temp, dim=-1)
            
            # Sample x0 based on importance weights
            x0_index = torch.multinomial(importance_weights, 1).view(x.shape[0])
            x0_samples = x0_samples.view(x.shape[0], k, -1)
            x0 = x0_samples[torch.arange(x.shape[0]), x0_index]
            
            # Cache the selected x0 as one-hot
            p_x0_cache = F.one_hot(x0, num_classes=self.vocab_size).float()
            _, x_next = self._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
        
        # Disable caching if state changed
        if (not torch.allclose(x_next, x) or self.time_conditioning):
          p_x0_cache = None
        x = x_next
      else:
        raise ValueError(f'Unknown sampler: {self.sampler}')

    # Final denoising step
    if self.config.sampling.noise_removal:
      t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
      unet_conditioning = self.noise(t)[0]
      x = self.forward(x, unet_conditioning).argmax(dim=-1)
    
    return x

  def _forward_pass_diffusion(self, x0, attention_mask=None, prefix=None):
    """Training forward pass using invariant energy function."""
    
    t = self._sample_t(x0.shape[0], x0.device)
    if self.T > 0:
      t = (t * self.T).to(torch.int)
      t = t / self.T
      t += (1 / self.T)

    if self.change_of_variables:
      unet_conditioning = t[:, None]
      f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
      f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
      move_chance = torch.exp(f_0 + t * (f_T - f_0))
      move_chance = move_chance[:, None]
    else:
      sigma, dsigma = self.noise(t)
      unet_conditioning = sigma[:, None]
      move_chance = 1 - torch.exp(-sigma[:, None])

    # Compute intermediate time s
    # s_ratio = self.intermediate_time_ratio
    if self.intermediate_time_ratio > 0:
      s_ratio = self.intermediate_time_ratio
    else:
      if prefix == 'train':
        s_ratio = random.random()
        # s_ratio = random.uniform(0, 0.8)
      else:
        s_ratio = random.random()
        # s_ratio = random.uniform(0, 0.8)
    s = t * s_ratio
    
    sigma_s, _ = self.noise(s)
    if self.change_of_variables:
      move_chance_s = torch.exp(f_0 + s * (f_T - f_0))
      move_chance_s = move_chance_s[:, None]
    else:
      move_chance_s = 1 - torch.exp(-sigma_s[:, None])
    
    # =========================================================================
    # Fix 1: Sample xs and xt with proper relationship: x0 -> xs -> xt
    # The diffusion process goes from clean (x0) to noisy (xt), with xs in between.
    # If a position is unmasked in xt, it must be unmasked in xs.
    # =========================================================================
    
    # First, sample xs from x0 (intermediate noisy state at time s)
    xs = self.q_xt(x0, move_chance_s)
    
    # Then, sample xt from xs by adding more masks
    # P(masked in xt | unmasked in xs) = (move_chance_t - move_chance_s) / (1 - move_chance_s)
    # This ensures that unmasked positions in xt are always unmasked in xs
    conditional_mask_prob = (move_chance - move_chance_s) / (1 - move_chance_s + 1e-8)
    additional_mask_indices = torch.rand(*xs.shape, device=xs.device) < conditional_mask_prob
    # Only mask positions that are currently unmasked in xs
    xt = torch.where(
      (xs != self.mask_index) & additional_mask_indices,
      torch.full_like(xs, self.mask_index),
      xs)
    
    # =========================================================================
    # Fix 2: Compute log p(xs|xt) properly using the transition kernel
    # For DDPM transition from xt (time t) to xs (time s < t):
    # - If xt[i] is unmasked: xs[i] = xt[i] deterministically
    # - If xt[i] is masked: 
    #   - P(xs[i]=v|xt) ∝ p_theta(x0=v|xt) * (t - s), for vocab tokens v
    #   - P(xs[i]=mask|xt) ∝ s
    #   - Normalization: Z = (t - s) + s = t
    # =========================================================================
    
    # Get diffusion model predictions
    with torch.no_grad():
      log_p_x0_given_xt = self.forward(xt, unet_conditioning).detach()

    # Positive samples: use ground truth x0 and xs
    x0_pos = x0
    xs_pos = xs
    
    # Negative samples: sample from model distributions
    k = 1
    # Sample x0_neg from original log_p_x0_given_xt (not modified)
    x0_neg = _sample_categorical(log_p_x0_given_xt.exp(), num_samples=k)
    
    # For xs_neg, we need to sample from the transition distribution
    # First compute log_p_xs for sampling (need proper transition probabilities)
    # Use the DDPM transition kernel for sampling
    # Ensure move_chance tensors have shape (batch, 1, 1) for broadcasting
    if move_chance.ndim == 2:
      move_chance_t_3d = move_chance[:, :, None]  # (batch, 1, 1)
    else:
      move_chance_t_3d = move_chance[:, None, None]  # (batch, 1, 1)
    
    if move_chance_s.ndim == 2:
      move_chance_s_3d = move_chance_s[:, :, None]  # (batch, 1, 1)
    else:
      move_chance_s_3d = move_chance_s[:, None, None]  # (batch, 1, 1)
    
    ################# 
    ### Please consider wheather use the sampled xs for negative energy calculation or not.
    ### If not, it will only calculate the negative energy based on the sampled x0 like the EBM class.
    ### If yes, it will calculate the negative energy based on the sampled xs like the invariant EBM class.
    ### Here is the code for sampling the xs.
    #################
    q_xs_for_sampling = log_p_x0_given_xt.exp() * (move_chance_t_3d - move_chance_s_3d)
    # Set mask token probability: need to broadcast to (batch, seq_len)
    q_xs_for_sampling[:, :, self.mask_index] = move_chance_s_3d[:, :, 0].expand(-1, q_xs_for_sampling.shape[1])
    xs_neg = _sample_categorical(q_xs_for_sampling, num_samples=k)
    
    # Ensure xs_neg copies unmasked tokens from xt (deterministic transition)
    copy_flag = (xt != self.mask_index).to(xt.dtype)
    xs_neg = copy_flag.repeat(k, 1) * xt.repeat(k, 1) + (1 - copy_flag.repeat(k, 1)) * xs_neg
    
    # Compute log_p_xs_given_xt for positive and negative samples SEPARATELY
    # (since they depend on the specific xs values)
    log_p_xs_pos_given_xt = self._compute_log_p_xs_given_xt(
      log_p_x0_given_xt, xt, xs_pos)
    log_p_xs_neg_given_xt = self._compute_log_p_xs_given_xt(
      log_p_x0_given_xt.repeat(k, 1, 1), xt.repeat(k, 1), xs_neg)
    
    is_eval = (prefix == 'val' or prefix == 'test')
    # is_eval = True
    energy_pos, term1_pos, term2_pos, term2_pos_x0_given_xt, term2_pos_ar_x0_given_xs = self.ebm_forward_invariant(
      xt=xt,
      xs=xs_pos,
      sigma_t=unet_conditioning,
      sigma_s=sigma_s[:, None] if not self.change_of_variables else None,
      x0=x0_pos,
      log_p_x0_given_xt=log_p_x0_given_xt,
      log_p_xs_given_xt=log_p_xs_pos_given_xt,
      attention_mask=attention_mask,
      return_terms=True,
      eval_mode=is_eval)
    
    ################# 
    ### Please consider wheather use the sampled xs for negative energy calculation or not.
    ### If not, it will only calculate the negative energy based on the sampled x0 like the EBM class.
    ### If yes, it will calculate the negative energy based on the sampled xs like the invariant EBM class.
    ### Here is the code for the sampled xs for negative energy calculation.
    ### xs = xs_pos or xs_neg? original code is xs_neg.
    #################
    energy_neg, term1_neg, term2_neg, term2_neg_x0_given_xt, term2_neg_ar_x0_given_xs = self.ebm_forward_invariant(
      xt=xt.repeat(k, 1),
      xs=xs_neg,
      sigma_t=unet_conditioning.repeat(k, 1),
      sigma_s=sigma_s[:, None].repeat(k, 1) if not self.change_of_variables else None,
      x0=x0_neg,
      log_p_x0_given_xt=log_p_x0_given_xt.repeat(k, 1, 1),
      log_p_xs_given_xt=log_p_xs_neg_given_xt,
      attention_mask=attention_mask.repeat(k, 1),
      return_terms=True,
      eval_mode=is_eval)
    energy_neg = energy_neg.view(x0.shape[0], k, -1)
    term1_neg = term1_neg.view(x0.shape[0], k, -1)
    term2_neg = term2_neg.view(x0.shape[0], k, -1)
    term2_neg_x0_given_xt = term2_neg_x0_given_xt.view(x0.shape[0], k, -1)
    term2_neg_ar_x0_given_xs = term2_neg_ar_x0_given_xs.view(x0.shape[0], k, -1)
    
    energy_neg = energy_neg[:, 0]

    # energy_sample, term1_sample, term2_sample, term2_sample_x0_given_xt, term2_sample_ar_x0_given_xs = self.ebm_forward_invariant(
    #   xt=xt,
    #   xs=xs_pos,
    #   sigma_t=unet_conditioning,
    #   sigma_s=sigma_s[:, None] if not self.change_of_variables else None,
    #   x0=x0_neg,
    #   log_p_x0_given_xt=log_p_x0_given_xt,
    #   log_p_xs_given_xt=log_p_xs_pos_given_xt,
    #   attention_mask=attention_mask,
    #   return_terms=True)

    model_output = torch.cat([energy_pos, energy_neg], dim=0)
    utils.print_nans(model_output, 'model_output')

    assert self.parameterization == 'subs'

    if prefix == 'train':
      # Noise contrastive estimation loss
      loss = - (torch.log(torch.sigmoid(-energy_pos) + 1e-8) \
                + torch.log(torch.sigmoid(energy_neg) + 1e-8))
      
      assert loss.shape[-1] == 1 and loss.ndim == 2
      return loss
    
    elif prefix == 'val' or prefix == 'test':
      # NLL Estimation (same as parent EBM class)
      if self.T == 0:
        log_p_theta = torch.gather(
          input=log_p_x0_given_xt,
          dim=-1,
          index=x0[:, :, None]).squeeze(-1)
      elif self.T > 0:
        diffusion_loss = self._d3pm_loss(
          model_output=log_p_x0_given_xt, xt=xt, x0=x0, t=t)
        if self.change_of_variables or self.importance_sampling:
          log_p_theta = diffusion_loss / torch.log1p(
            - torch.exp(- self.noise.sigma_min))
        else:
          log_p_theta = - diffusion_loss / (
            dsigma / torch.expm1(sigma))[:, None]
      else:
        raise ValueError(f'Unknown T: {self.T}')

      # ============================================
      # EBM correction for InvariantEBM
      # 
      # Six modes controlled by partition_mode:
      #   "analytical"             -> log Z = -term1 exactly (zero variance)
      #                               log_p_phi = -term2_pos / seq_len
      #   "from_x0"                -> Empirical: vary x0, fix xs (full energy)
      #                               log_p_phi = (-energy_pos - log_Z) / seq_len
      #   "from_xs"                -> Empirical: fix x0, vary xs
      #                               log_p_phi = (-energy_pos - log_Z) / seq_len
      #   "from_definition"        -> Eq.(27): x0 ~ p_theta(x0|xt)
      #                               score = -term2 = log p(x0|xt) - log P_AR(x0|xs)
      #   "from_def_diffusion_xs"  -> Eq.(27): x0 ~ p_theta(x0|xs), IS-weighted
      #                               score = 2*log p(x0|xt) - log p(x0|xs) - log P_AR(x0|xs)
      #   "from_def_ar_xs"         -> Eq.(27): x0 ~ p_AR(x0|xs), IS-weighted
      #                               score = 2*(log p(x0|xt) - log P_AR(x0|xs))
      # ============================================
      partition_mode = getattr(
        self.config.sampling, 'partition_mode', 'analytical')
      n_partition_samples = getattr(
        self.config.sampling, 'n_partition_samples', 16)
      use_loo = getattr(
        self.config.sampling, 'use_leave_one_out', True)

      if partition_mode == 'from_x0' and n_partition_samples > 0:
        # Empirical: vary x0, fix xs, with real x0 as anchor
        log_Z = self._estimate_log_partition_function(
          xt=xt,
          unet_conditioning=unet_conditioning,
          log_p_x0_given_xt=log_p_x0_given_xt,
          x0_fixed=x0_pos,
          xs_fixed=xs_pos,
          log_p_xs_fixed_given_xt=log_p_xs_pos_given_xt,
          sigma_s=sigma_s,
          attention_mask=attention_mask,
          n_samples=n_partition_samples,
          use_leave_one_out=use_loo)
        log_p_phi = (- energy_pos - log_Z) / log_p_theta.shape[-1]
      elif partition_mode == 'from_xs' and n_partition_samples > 0:
        # Empirical: fix x0, vary xs, with xs_pos as anchor
        log_Z = self._estimate_log_partition_function_from_xs(
          xt=xt,
          unet_conditioning=unet_conditioning,
          log_p_x0_given_xt=log_p_x0_given_xt,
          x0_fixed=x0_pos,
          xs_pos=xs_pos,
          move_chance_s=move_chance_s,
          move_chance_t=move_chance,
          sigma_s=sigma_s,
          attention_mask=attention_mask,
          n_samples=n_partition_samples,
          use_leave_one_out=use_loo)
        log_p_phi = (- energy_pos - log_Z) / log_p_theta.shape[-1]
      elif partition_mode == 'from_definition' and n_partition_samples > 0:
        # Eq.(27): x0 ~ p_theta(x0|xt), score = -term2
        log_Z = self._estimate_log_partition_function_from_definition(
          xt=xt,
          log_p_x0_given_xt=log_p_x0_given_xt,
          x0_pos=x0_pos,
          xs_pos=xs_pos,
          term1_pos=term1_pos,
          attention_mask=attention_mask,
          n_samples=n_partition_samples,
          use_leave_one_out=use_loo)
        log_p_phi = (- energy_pos - log_Z) / log_p_theta.shape[-1]
      elif partition_mode == 'from_def_diffusion_xs' and n_partition_samples > 0:
        # Eq.(27) with x0 sampled from diffusion model conditioned on xs
        log_Z = self._estimate_log_Z_from_definition_diffusion_xs(
          xt=xt,
          log_p_x0_given_xt=log_p_x0_given_xt,
          x0_pos=x0_pos,
          xs_pos=xs_pos,
          sigma_s=sigma_s,
          term1_pos=term1_pos,
          attention_mask=attention_mask,
          n_samples=n_partition_samples,
          use_leave_one_out=use_loo)
        log_p_phi = (- energy_pos - log_Z) / log_p_theta.shape[-1]
        # log_p_phi_term2 = - term2_pos / log_p_theta.shape[-1]
        # log_p_phi_ori = - energy_pos / log_p_theta.shape[-1]
        # print(f"log_p_phi: {log_p_phi}")
        # print(f"log_p_phi_term2: {log_p_phi_term2}")
        # print(f"log_p_phi_ori: {log_p_phi_ori}")
        # print("--------------------------------")
        # print(f"energy_pos: {energy_pos}")
        # print(f"log_Z: {log_Z}")
        # print(f"energy_pos + log_Z: {energy_pos + log_Z}")
        # print("--------------------------------")
        # exit(0)
      elif partition_mode == 'from_def_ar_xs' and n_partition_samples > 0:
        # Eq.(27) with x0 sampled from AR model conditioned on xs
        log_Z = self._estimate_log_Z_from_definition_ar_xs(
          xt=xt,
          log_p_x0_given_xt=log_p_x0_given_xt,
          x0_pos=x0_pos,
          xs_pos=xs_pos,
          term1_pos=term1_pos,
          attention_mask=attention_mask,
          n_samples=n_partition_samples,
          use_leave_one_out=use_loo)
        log_p_phi = (- energy_pos - log_Z) / log_p_theta.shape[-1]
        # log_p_phi_term2 = - term2_pos / log_p_theta.shape[-1]
        # log_p_phi_ori = - energy_pos / log_p_theta.shape[-1]
        # print(f"log_p_phi: {log_p_phi}")
        # print(f"log_p_phi_term2: {log_p_phi_term2}")
        # print(f"log_p_phi_ori: {log_p_phi_ori}")
        # print("--------------------------------")
        # print(f"energy_pos: {energy_pos}")
        # print(f"log_Z: {log_Z}")
        # print(f"energy_pos + log_Z: {energy_pos + log_Z}")
        # print("--------------------------------")
        # exit(0)
      elif partition_mode == 'analytical':
        # Analytical (default): log Z = -term1, so -E_pos - log Z = -term2_pos
        # log_p_phi = term2_pos_ar_x0_given_xs / log_p_theta.shape[-1]
        log_p_phi = - term2_pos / log_p_theta.shape[-1]
      else:
        log_p_phi = - energy_pos / log_p_theta.shape[-1]
      assert log_p_theta.ndim == log_p_phi.ndim
      log_p = log_p_theta + log_p_phi
      # log_p = log_p_phi
      
      # if attention_mask is None:
      #   tmp_attention_mask = torch.ones_like(x0)
      # else:
      #   tmp_attention_mask = attention_mask
      # parameterization = self.parameterization
      # self.parameterization = 'ar'
      # (x0_input_tokens, x0_output_tokens, _) = self._maybe_sub_sample(x0, tmp_attention_mask)
      # (xs_input_tokens, xs_output_tokens, _) = self._maybe_sub_sample(xs, tmp_attention_mask)
      # (xt_input_tokens, xt_output_tokens, _) = self._maybe_sub_sample(xt, tmp_attention_mask)    
      # self.parameterization = parameterization

      # with torch.cuda.amp.autocast(dtype=torch.bfloat16):
      #   logits_ar_given_xt = self._compute_ar_logits(x0_input_tokens, xt_output_tokens)
      #   log_p_ar_xs_given_xt = (logits_ar_given_xt.gather(-1, x0_output_tokens[:, :, None])[:, :, 0]).sum(dim=-1, keepdim=True)
      
      # log_p = log_p_ar_xs_given_xt / log_p_theta.shape[-1]

      if self.change_of_variables or self.importance_sampling:
        return log_p * torch.log1p(- torch.exp(- self.noise.sigma_min))
      
      return - log_p * (dsigma / torch.expm1(sigma))[:, None]
    
    else:
      raise ValueError(f'Unknown prefix: {prefix}')
