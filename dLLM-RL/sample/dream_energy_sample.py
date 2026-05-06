"""Dream sampling with Energy-Based Model reranking.

Drop-in replacement for ``dream_sample.py``.  At each denoising step the
script samples *k* sets of x0 candidates from the diffusion model, scores
them with a pretrained AR model, and selects the best candidate before
applying the standard confidence-based unmasking.

Two energy methods are supported via ``energy.energy_method``:
  * ``default``  – rerank full x0 candidates at all masked positions.
  * ``adaption`` – for each candidate select the top-*m* positions by DLLM
    confidence, score only those positions, and use the best partial unmask.
"""
from __future__ import annotations
import math, json, os, time, re, random, types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from jinja2 import Template
import torch
import torch.nn.functional as F
import torch.distributions as dists
from termcolor import cprint
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers.utils import ModelOutput
import multiprocessing as mp

from dream import DreamTokenizer
from dream.modeling_dream import DreamModel
from dream.generation_utils_block import DreamGenerationMixin, DreamGenerationConfig

from omegaconf import DictConfig, ListConfig, OmegaConf

from energy_utils import (
    load_ar_model,
    energy_rerank,
    energy_rerank_adaption,
    build_energy_config,
    EnergyConfig,
)


# ──────────────────────────── config ────────────────────────────────────────

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


# ──────────────────────────── helpers ───────────────────────────────────────

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, tar=None):
    logits = logits.float()
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    dist = dists.Categorical(logits=logits)
    x0 = dist.sample()
    probs = dist.probs
    if temperature > 0:
        target = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        target, x0 = probs.max(dim=-1)
    if tar == "confidence":
        return target, x0
    if tar == "margin_confidence":
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        target = sorted_probs[:, 0] - sorted_probs[:, 1]
    if tar == "neg_entropy":
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        target = torch.sum(probs * log_probs, dim=-1)
    return target, x0


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


# ──────────────────────────── generation loop ──────────────────────────────

@torch.no_grad()
def _sample_with_energy(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    block_length: Optional[int] = 32,
    use_cache: bool = False,
    further_horizon: int = 128,
    mask_token_id: int = 151666,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    pad_target_penalty: float = 1.0,
    unmask_threshold: float = 0.9,
    ar_model=None,
    ecfg: EnergyConfig = None,
) -> Union[DreamModelOutput, torch.LongTensor]:

    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    steps = generation_config.steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon

    histories = [] if (return_dict_in_generate and output_history) else None

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    gen_length = max_length - input_ids.shape[1]

    if block_length is None:
        block_length = gen_length
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    base_s, rem = divmod(steps, num_blocks)
    steps_per_block = [base_s + (1 if i < rem else 0) for i in range(num_blocks)]
    timesteps = [
        torch.linspace(1, generation_config.eps, spb + 1, device=x.device)
        for spb in steps_per_block
    ]

    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attention_mask = F.pad(attention_mask,
                               (0, max_length - attention_mask.shape[1]),
                               value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        attention_mask = torch.logical_and(
            attention_mask.unsqueeze(1).unsqueeze(-2),
            attention_mask.unsqueeze(1).unsqueeze(-1))
        attention_mask = torch.where(
            attention_mask,
            torch.tensor(0.0, device=attention_mask.device),
            torch.tensor(float("-inf"), device=attention_mask.device))
    else:
        tok_idx = None
        attention_mask = "full"

    past_key_values = None
    energy_step_counter = 0
    estimated_total_iters = max(steps, gen_length) if unmask_threshold is not None else steps
    energy_cutoff = int(estimated_total_iters * ecfg.energy_fraction) if ecfg is not None else 0
    is_adaption = (ecfg is not None and ecfg.energy_method == "adaption")

    for num_block in range(num_blocks):
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end   = current_block_start + block_length

        if cgws is not None:
            window_end   = min(current_block_end + cgws, max_length)
            window_slice = slice(current_block_start, window_end)

        # ── update KV cache ──
        if use_cache:
            model_output = model(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            new_pkv = []
            for i in range(len(past_key_values)):
                new_pkv.append(())
                for j in range(len(past_key_values[i])):
                    new_pkv[i] += (past_key_values[i][j][:, :current_block_start, :],)
            past_key_values = new_pkv
        else:
            model_output = model(x, attention_mask, tok_idx, use_cache=False)

        logits = model_output.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        _, x0_first = sample_tokens(logits, temperature=temperature,
                                     top_p=top_p, top_k=top_k)
        x[:, current_block_start] = x0_first[:, current_block_start]
        if histories is not None:
            histories.append(x.clone().cpu())

        spb = steps_per_block[num_block]
        i_step = 1
        while True:
            if cgws is not None:
                mask_index = (x[:, window_slice] == mask_token_id)
            else:
                mask_index = (x[:, current_block_start:] == mask_token_id)

            if attention_mask != "full":
                if cgws is not None:
                    current_attention_mask = attention_mask[:, :, window_slice, :window_end]
                else:
                    current_attention_mask = attention_mask[:, :, current_block_start:, :]
            else:
                current_attention_mask = attention_mask

            if use_cache:
                if cgws is not None:
                    mo = model(x[:, window_slice], current_attention_mask,
                               tok_idx[:, window_slice] if tok_idx is not None else None,
                               past_key_values=past_key_values, use_cache=True)
                else:
                    mo = model(x[:, current_block_start:], current_attention_mask,
                               tok_idx[:, current_block_start:] if tok_idx is not None else None,
                               past_key_values=past_key_values, use_cache=True)
                logits = mo.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            else:
                mo = model(x, attention_mask, tok_idx, use_cache=False)
                logits = mo.logits
                logits = logits[:, current_block_start:]
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break

            mask_index[:, block_length:] = False
            mask_logits = logits[mask_index]
            target, x0 = sample_tokens(mask_logits, temperature,
                                        top_p=top_p, top_k=top_k, tar=tar)

            # ── energy check ──
            use_energy = (ecfg is not None and ar_model is not None
                          and ecfg.energy_frequency > 0
                          and energy_step_counter < energy_cutoff
                          and energy_step_counter % ecfg.energy_frequency == 0)
            energy_step_counter += 1

            adaption_applied = False

            # ────────── adaption energy method ──────────
            if use_energy and is_adaption and mask_index[:, :block_length].any():
                B = x.shape[0]
                k_e = ecfg.is_size
                bl = block_length
                block_mask_2d = mask_index[:, :bl]
                num_masked = int(block_mask_2d.sum().item())

                if num_masked > 0:
                    # Sample k candidate sets (flat)
                    cands_flat = [x0.clone()]
                    confs_flat = [target.clone()]
                    for _ in range(k_e - 1):
                        t_k, x0_k = sample_tokens(
                            mask_logits, temperature,
                            top_p=top_p, top_k=top_k, tar=tar)
                        cands_flat.append(x0_k)
                        confs_flat.append(t_k)

                    # Reconstruct (B, k, bl) block candidates and confidences
                    xwin = (x[:, window_slice] if cgws is not None
                            else x[:, current_block_start:])
                    block_cands = xwin[:, :bl].unsqueeze(1).expand(
                        B, k_e, bl).clone()
                    block_confs = torch.full(
                        (B, k_e, bl), float('-inf'), device=x.device)

                    for ci in range(k_e):
                        filled = xwin[:, :bl].clone()
                        filled[block_mask_2d] = cands_flat[ci]
                        block_cands[:, ci] = filled
                        conf_tmp = torch.full(
                            (B, bl), float('-inf'), device=x.device)
                        conf_tmp[block_mask_2d] = confs_flat[ci]
                        block_confs[:, ci] = conf_tmp

                    # Determine m
                    m_val = ecfg.adaption_m
                    if m_val <= 0:
                        num_mask_token = mask_index.sum() / B
                        if i_step < spb:
                            t_val = timesteps[num_block][i_step].item()
                            s_val = timesteps[num_block][i_step + 1].item()
                            m_val = max(1, int(
                                num_mask_token * (1 - s_val / t_val))
                                if i_step < spb - 1
                                else int(num_mask_token))
                        else:
                            m_val = max(1, int(num_mask_token))

                    selected_block, transfer_mask = energy_rerank_adaption(
                        x, block_cands, block_confs, block_mask_2d,
                        current_block_start, current_block_end,
                        ar_model, mask_token_id, ecfg, m_val)

                    # Apply the partial unmask directly
                    x[:, current_block_start:current_block_end][
                        transfer_mask] = selected_block[transfer_mask]
                    adaption_applied = True

            # ────────── default energy method ──────────
            elif use_energy and mask_index[:, :block_length].any():
                B = x.shape[0]
                k_e = ecfg.is_size
                bl = block_length
                block_mask = mask_index[:, :bl]
                num_masked = int(block_mask.sum().item())

                if num_masked > 0:
                    cands_list = [x0.clone()]
                    for _ in range(k_e - 1):
                        _, x0_extra = sample_tokens(
                            mask_logits, temperature,
                            top_p=top_p, top_k=top_k, tar=tar)
                        cands_list.append(x0_extra)

                    xwin = (x[:, window_slice] if cgws is not None
                            else x[:, current_block_start:])
                    block_cands = torch.zeros(
                        B, k_e, bl, device=x.device, dtype=torch.long)
                    for ci, c_x0 in enumerate(cands_list):
                        tmp = torch.full_like(
                            xwin[:, :bl], mask_token_id)
                        tmp[block_mask] = (c_x0[:num_masked]
                                           if ci == 0 else cands_list[ci])
                        filled = xwin[:, :bl].clone()
                        filled[block_mask] = tmp[block_mask]
                        block_cands[:, ci, :] = filled

                    log_p_block = None
                    if ecfg.energy_type == "invariant":
                        log_p_block = F.log_softmax(
                            logits[:, :bl].float(), dim=-1)

                    best_block, sel_idx = energy_rerank(
                        x, block_cands, block_mask,
                        current_block_start, current_block_end,
                        ar_model, mask_token_id, ecfg, log_p_block)

                    selected_flat = best_block[block_mask]
                    x0 = selected_flat

                    # Recompute confidence for the selected tokens
                    probs_flat = F.softmax(mask_logits.float(), dim=-1)
                    if tar == "confidence":
                        target = probs_flat.gather(
                            -1, x0.unsqueeze(-1)).squeeze(-1)
                    elif tar == "margin_confidence":
                        sorted_p, _ = torch.sort(
                            probs_flat, dim=-1, descending=True)
                        target = sorted_p[:, 0] - sorted_p[:, 1]
                    elif tar == "neg_entropy":
                        target = torch.sum(
                            probs_flat * torch.log(probs_flat + 1e-10),
                            dim=-1)
                    else:
                        target = probs_flat.gather(
                            -1, x0.unsqueeze(-1)).squeeze(-1)

            # ────────── standard unmasking (skip if adaption applied) ──────
            if not adaption_applied:
                # Pad token penalty
                _pad_mask_flat = (x0 == pad_token_id)
                if _pad_mask_flat.any():
                    target = target.clone()
                    target[_pad_mask_flat] = (
                        target[_pad_mask_flat] / pad_target_penalty)

                if cgws is not None:
                    full_target = torch.full_like(
                        x[:, window_slice], -torch.inf,
                        device=model.device, dtype=logits.dtype)
                else:
                    full_target = torch.full_like(
                        x[:, current_block_start:], -torch.inf,
                        device=model.device, dtype=logits.dtype)
                full_target = full_target.float()
                full_target[mask_index] = target
                full_target[:, block_length:] = -torch.inf

                if unmask_threshold is None:
                    num_mask_token = mask_index.sum() / mask_index.shape[0]
                    t = timesteps[num_block][i_step]
                    s = timesteps[num_block][i_step + 1]
                    number_transfer_tokens = (
                        int(num_mask_token * (1 - s / t))
                        if i_step < spb - 1
                        else int(num_mask_token))

                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(
                                full_target, number_transfer_tokens)
                        else:
                            ft = F.softmax(full_target / alg_temp, dim=-1)
                            transfer_index = torch.multinomial(
                                ft, num_samples=number_transfer_tokens)

                        if cgws is not None:
                            x_ = (torch.zeros_like(
                                x[:, window_slice],
                                device=model.device, dtype=torch.long)
                                + mask_token_id)
                        else:
                            x_ = (torch.zeros_like(
                                x[:, current_block_start:],
                                device=model.device, dtype=torch.long)
                                + mask_token_id)
                        x_[mask_index] = x0.clone()
                        row_indices = (
                            torch.arange(x.size(0), device=model.device)
                            .unsqueeze(1).expand_as(transfer_index))
                        if cgws is not None:
                            x[:, window_slice][
                                row_indices, transfer_index] = \
                                x_[row_indices, transfer_index]
                        else:
                            x[:, current_block_start:][
                                row_indices, transfer_index] = \
                                x_[row_indices, transfer_index]
                else:
                    xwin = (x[:, window_slice] if cgws is not None
                            else x[:, current_block_start:])
                    selected_map = torch.zeros_like(
                        xwin, dtype=torch.bool)
                    selected_map[mask_index] = (target >= unmask_threshold)
                    no_sel = (~selected_map.any(dim=-1)
                              & mask_index.any(dim=-1))
                    if no_sel.any():
                        masked_scores = full_target.masked_fill(
                            ~mask_index, float("-inf"))
                        best_idx = torch.argmax(masked_scores, dim=-1)
                        sel_rows = torch.nonzero(
                            no_sel, as_tuple=False).squeeze(-1)
                        selected_map[sel_rows, best_idx[sel_rows]] = True
                    selected_map &= mask_index
                    x_candidates = torch.full_like(
                        xwin, mask_token_id, dtype=torch.long)
                    x_candidates[mask_index] = x0
                    xwin[selected_map] = x_candidates[selected_map]

            if histories is not None:
                histories.append(x.clone().cpu())

            i_step += 1

            if (x[:, current_block_start:current_block_end]
                    == mask_token_id).sum() == 0:
                break

        block_all_pad = torch.all(
            x[:, current_block_start:current_block_end] == pad_token_id)
        if block_all_pad:
            if current_block_end < x.size(1):
                x[:, current_block_end:] = pad_token_id
            if histories is not None:
                histories.append(x.clone().cpu())
            break

    if return_dict_in_generate:
        return DreamModelOutput(sequences=x, history=histories)
    return x


# ──────────────────────────── utility functions ────────────────────────────

def random_select(data_list, random_k):
    return random.sample(data_list, random_k)


def get_prompt(data_i):
    return Template(system_prompts).render(problem=data_i["question"])


def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"
    i = start + len(tag)
    depth = 1
    buf = []
    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    return matches[-1].strip() if matches else "We can not extract the code in the output. "


def denoise_step_map(history, mask_id: int, sample_idx: int = 0):
    L = history[0].shape[1]
    step_map = torch.zeros(L, dtype=torch.long)
    prev = torch.full((L,), mask_id, dtype=torch.long)
    for t, snap in enumerate(history, start=0):
        cur = snap[sample_idx]
        changed = (prev == mask_id) & (cur != mask_id)
        step_map[changed] = t
        prev = cur
        if (step_map == 0).sum() == 0:
            break
    return step_map


def get_data_chunk(data, num_node, node_idx):
    total = len(data)
    chunk_size = (total + num_node - 1) // num_node
    start_idx = node_idx * chunk_size
    end_idx = min((node_idx + 1) * chunk_size, total)
    return data[start_idx:end_idx]


def get_token_lengths(strings, tokenizer):
    pad_token = tokenizer.pad_token
    escaped = re.escape(pad_token)
    pattern = rf"(?:{escaped})+"
    collapse_re = re.compile(pattern)
    lengths = []
    for s in strings:
        s_clean = collapse_re.sub(
            lambda _: pad_token if isinstance(pad_token, str) else '', s)
        s_clean = re.sub(escaped, '', s_clean)
        lengths.append(len(tokenizer.encode(s_clean, add_special_tokens=False)))
    return lengths


# ──────────────────────────── GPU worker ───────────────────────────────────
from tqdm import tqdm


def worker(pretrained_model, ar_model_path, rank, prompts, orig_idx,
           seq_dict, step_dict, batch_size, config, ecfg_dict):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model_gpu = (DreamModel.from_pretrained(pretrained_model,
                                            trust_remote_code=True,
                                            torch_dtype=torch.bfloat16)
                 .to(device).eval())
    model_gpu.diffusion_generate = types.MethodType(
        DreamGenerationMixin.diffusion_generate, model_gpu)
    model_gpu._sample = types.MethodType(
        DreamGenerationMixin._sample, model_gpu)

    ecfg = EnergyConfig(**ecfg_dict)
    ar_model = load_ar_model(
        ar_model_path, device,
        load_in_4bit=ecfg.ar_load_in_4bit,
        load_in_8bit=ecfg.ar_load_in_8bit)

    tokenizer_gpu = DreamTokenizer.from_pretrained(
        pretrained_model, trust_remote_code=True)

    pad_id  = model_gpu.config.pad_token_id
    mask_id = model_gpu.config.mask_token_id
    eos_id  = tokenizer_gpu.convert_tokens_to_ids("<|im_end|>")

    for start in tqdm(range(0, len(prompts), batch_size),
                      desc=f"GPU {rank}", position=rank, leave=True):
        batch_prompts = prompts[start:start + batch_size]
        batch_idxs    = orig_idx[start:start + batch_size]

        enc = tokenizer_gpu(batch_prompts, padding=True,
                            return_tensors="pt", padding_side="left")
        prompt_ids = enc["input_ids"].to(device)
        attn_mask  = prompt_ids.ne(pad_id).to(device=model_gpu.device)

        fh = config.rollout.further_horizon
        if not config.rollout.use_cache:
            fh = None

        gen_cfg = DreamGenerationConfig(
            output_history=True,
            return_dict_in_generate=True,
            max_length=config.rollout.max_gen_length + prompt_ids.shape[1],
            steps=config.rollout.steps,
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            top_k=config.rollout.top_k,
            tar=config.rollout.target,
            alg_temp=config.rollout.alg_temp,
        )

        unmask_threshold = (None
                            if config.rollout.remasking_strategy == "low_confidence_static"
                            else config.rollout.dynamic_threshold)

        generation_ids = _sample_with_energy(
            model_gpu, prompt_ids,
            attention_mask=attn_mask,
            generation_config=gen_cfg,
            block_length=config.rollout.block_size,
            use_cache=config.rollout.use_cache,
            further_horizon=fh,
            mask_token_id=mask_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            pad_target_penalty=config.rollout.pad_target_penalty,
            unmask_threshold=unmask_threshold,
            ar_model=ar_model,
            ecfg=ecfg,
        )
        generation_ids.sequences = generation_ids.sequences.cpu()
        torch.cuda.empty_cache()

        seq_ids = generation_ids.sequences[:, prompt_ids.shape[1]:].tolist()
        texts = tokenizer_gpu.batch_decode(
            seq_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=True)

        for i, idx in enumerate(batch_idxs):
            m = denoise_step_map(generation_ids.history,
                                 mask_id=mask_id, sample_idx=i)
            step_map = m[prompt_ids.shape[1]:].tolist()
            seq_dict[idx]  = texts[i]
            step_dict[idx] = step_map

        torch.cuda.empty_cache()


# ──────────────────────────── main ─────────────────────────────────────────
if __name__ == "__main__":

    config = get_config()
    mp.set_start_method("spawn", force=True)

    k_sample   = config.rollout.num_response_per_task
    batch_size = config.rollout.batch_size

    project_name = config.experiment.project
    ecfg = build_energy_config(config.energy)
    ecfg_dict = ecfg.__dict__

    system_prompts = (
        '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'''
        '''<|im_start|>user\nYou need to put your final answer in \\boxed{}. '''
        '''This is the problem:\n{{problem}}<|im_end|>\n'''
        '''<|im_start|>assistant\n'''
    )

    code_eval = False

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model
    if config.dataset.data_type == "code":
        code_eval = True
        system_prompts_function = (
            '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'''
            '''<|im_start|>user\n{{problem}}\nPlace your code within a single Python '''
            '''code block ```python ```. Do not include more than one code block. '''
            '''<|im_end|>\n<|im_start|>assistant\n''')
        system_prompts_stdio = (
            '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'''
            '''<|im_start|>user\nThis is the problem:\n{{problem}}\n'''
            '''You should put your code in ```python ```. '''
            '''Use input() to read input and print() to produce output in your script. '''
            '''<|im_end|>\n<|im_start|>assistant\n''')
    elif config.dataset.data_type == "option":
        system_prompts = (
            '''<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'''
            '''<|im_start|>user\nThis is the problem:\n{{problem}}\n'''
            '''You need to think step by step and put the final option '''
            '''(A, B, C, or D only—no other character) in \\boxed{}. '''
            '''<|im_end|>\n<|im_start|>assistant\n''')

    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    data_dir = OmegaConf.select(config, "dataset.data_dir", default="../data")
    with open(os.path.join(data_dir, dataset + ".json"), 'r') as f:
        data = json.load(f)

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if num_node > 1:
        data = get_data_chunk(data, num_node, node_index)

    num = len(data)
    tokenizer = DreamTokenizer.from_pretrained(
        pretrained_model, trust_remote_code=True)

    generation_prompts = []
    prefix_list = []
    index_list = []
    for i in range(num):
        if code_eval:
            if data[i]["test_method"] == "stdio":
                system_prompts = system_prompts_stdio
                prefix_list += [None] * k_sample
            else:
                system_prompts = system_prompts_function + data[i]["prefix"]
                prefix_list += [data[i]["prefix"]] * k_sample
        generation_prompts += [get_prompt(data[i])] * k_sample
        index_list += [i] * k_sample
        data[i]["full_output"] = []
        data[i]["step_map"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        data[i]["prompt"] = get_prompt(data[i])

    cprint("start generation (energy-enhanced Dream)...", "green")
    generation_start_time = time.time()

    all_prompts = generation_prompts
    N = len(all_prompts)
    shuffled_idx = list(range(N))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [all_prompts[i] for i in shuffled_idx]

    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "need >=1 GPU"

    def split_even(lst, n):
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

    prompt_chunks = split_even(shuffled_prompts, n_gpu)
    idx_chunks    = split_even(shuffled_idx, n_gpu)

    manager   = mp.Manager()
    seq_dict  = manager.dict()
    step_dict = manager.dict()
    procs = []

    for rk in range(n_gpu):
        p = mp.Process(
            target=worker,
            args=(pretrained_model, ecfg.ar_model_path, rk,
                  prompt_chunks[rk], idx_chunks[rk],
                  seq_dict, step_dict, batch_size, config, ecfg_dict))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    restored_outputs   = [seq_dict[i]  for i in range(N)]
    restored_step_maps = [step_dict[i] for i in range(N)]

    generation_elapsed_time = time.time() - generation_start_time
    cprint("generation job done!", "green")

    response_length = get_token_lengths(restored_outputs, tokenizer)
    mean_response_length = sum(response_length) / len(response_length)

    total_generated_tokens = sum(response_length)
    throughput = total_generated_tokens / generation_elapsed_time if generation_elapsed_time > 0 else 0.0
    cprint(f"Throughput: {throughput:.2f} tokens/s "
           f"(total_tokens={total_generated_tokens}, time={generation_elapsed_time:.2f}s)", "cyan")

    i = 0
    for full_output in restored_outputs:
        if code_eval:
            if data[int(i / k_sample)]["test_method"] == "function":
                extracted_output = extract_code(prefix_list[i] + full_output)
            else:
                extracted_output = extract_code(full_output)
        else:
            extracted_output = extract_final_boxed_answer(full_output)
        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        data[index_i]["step_map"].append(restored_step_maps[i])
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])
        i += 1

    if num_node > 1:
        output_file_name = (f"../{project_name}/temp_data/"
                            f"outputs-{node_index}-{outputs_name}.json")
    else:
        output_file_name = (f"../{project_name}/temp_data/"
                            f"outputs-{outputs_name}.json")
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    throughput_info = {
        "total_generated_tokens": total_generated_tokens,
        "generation_time_seconds": generation_elapsed_time,
        "throughput_tokens_per_second": throughput,
        "num_samples": N,
        "mean_response_length": mean_response_length,
    }
    throughput_file = output_file_name.replace("outputs-", "throughput-")
    with open(throughput_file, "w", encoding="utf-8") as f:
        json.dump(throughput_info, f, indent=2, ensure_ascii=False)
