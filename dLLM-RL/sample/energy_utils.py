"""Energy-Based Model utilities for dLLM-RL.

Uses a pretrained autoregressive (AR) model to compute energy scores that
guide the masked-diffusion sampling loop (LLaDa / Dream).

Energy methods
--------------
* ``default``   – sample *k* full x0 candidates, score every masked position
  with the AR model, pick one candidate via importance sampling.
* ``adaption``  – for each x0 candidate, select the top-*m* positions by
  DLLM confidence to unmask, score only those transition positions with the
  AR model, and deterministically pick the best partial unmask.

Within the *default* method two scoring modes are supported:
* ``ar_rerank``  – pure AR log-probability.
* ``invariant``  – AR score + diffusion invariant correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


# ───────────────────────────── Configuration ──────────────────────────────

@dataclass
class EnergyConfig:
    ar_model_path: str = ""
    is_size: int = 2                     # number of x0 candidates (k)
    is_temp: float = 1.0                 # importance-sampling temperature
    energy_type: str = "ar_rerank"       # "ar_rerank" | "invariant"
    energy_method: str = "default"       # "default" | "adaption"
    s_ratio: float = 0.5                 # invariant energy: mask-keep ratio
    adaption_m: int = -1                 # adaption unmask count (-1 = auto)
    ar_load_in_4bit: bool = False
    ar_load_in_8bit: bool = False
    energy_frequency: int = 1            # apply every N denoising steps
    energy_fraction: float = 1.0         # fraction of initial steps that use energy


def build_energy_config(cfg) -> EnergyConfig:
    """Build :class:`EnergyConfig` from an OmegaConf subtree."""
    return EnergyConfig(
        ar_model_path=str(getattr(cfg, "ar_model_path", "")),
        is_size=int(getattr(cfg, "is_size", 2)),
        is_temp=float(getattr(cfg, "is_temp", 1.0)),
        energy_type=str(getattr(cfg, "energy_type", "ar_rerank")),
        energy_method=str(getattr(cfg, "energy_method", "default")),
        s_ratio=float(getattr(cfg, "s_ratio", 0.5)),
        adaption_m=int(getattr(cfg, "adaption_m", -1)),
        ar_load_in_4bit=bool(getattr(cfg, "ar_load_in_4bit", False)),
        ar_load_in_8bit=bool(getattr(cfg, "ar_load_in_8bit", False)),
        energy_frequency=int(getattr(cfg, "energy_frequency", 1)),
        energy_fraction=float(getattr(cfg, "energy_fraction", 1.0)),
    )


# ───────────────────────────── AR model loading ───────────────────────────

def load_ar_model(model_path: str, device, dtype=torch.bfloat16, *,
                  load_in_4bit: bool = False, load_in_8bit: bool = False):
    """Load a HuggingFace causal LM for energy scoring."""
    kwargs: dict = dict(trust_remote_code=True)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        kwargs["device_map"] = {"": device}
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": device}
    else:
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if "device_map" not in kwargs:
        model = model.to(device)
    model.eval()
    return model


# ───────────────────────────── KV-cache helpers ───────────────────────────

def _repeat_kv_cache(past_key_values, repeats: int):
    """Replicate a KV cache *repeats* times along the batch dimension."""
    if past_key_values is None or repeats <= 1:
        return past_key_values

    # DynamicCache (transformers >= 4.36)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        try:
            from transformers.cache_utils import DynamicCache
            new = DynamicCache()
            new.key_cache = [
                k.repeat_interleave(repeats, dim=0)
                for k in past_key_values.key_cache
            ]
            new.value_cache = [
                v.repeat_interleave(repeats, dim=0)
                for v in past_key_values.value_cache
            ]
            if hasattr(past_key_values, "_seen_tokens"):
                new._seen_tokens = past_key_values._seen_tokens
            return new
        except ImportError:
            pass

    # Tuple-of-tuples (older transformers)
    if isinstance(past_key_values, (tuple, list)):
        return tuple(
            tuple(t.repeat_interleave(repeats, dim=0) for t in layer)
            for layer in past_key_values
        )
    raise TypeError(f"Unsupported KV cache type: {type(past_key_values)}")


# ──────────────────────── Low-level AR scoring ────────────────────────────

@torch.no_grad()
def compute_ar_log_probs(ar_model, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-position log p(token_i | tokens_{<i}) for i = 1..L-1.

    Returns ``(N, L-1)`` tensor of log-probabilities.
    """
    out = ar_model(input_ids=input_ids, use_cache=False)
    log_probs = F.log_softmax(out.logits.float(), dim=-1)
    return log_probs[:, :-1].gather(
        -1, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)


@torch.no_grad()
def _score_with_kv_cache(ar_model, safe, eval_masks, P, bl, B, k, device):
    """KV-cache accelerated block scoring (internal helper).

    The prefix ``safe[:, 0, :P]`` (shared across all *k* candidates) is
    processed once; only the differing block suffix is run *B·k* times.
    """
    prefix = safe[:, 0, :P]                                        # (B, P)
    prefix_out = ar_model(input_ids=prefix, use_cache=True)
    pkv = prefix_out.past_key_values
    prefix_last_lp = F.log_softmax(
        prefix_out.logits[:, -1:, :].float(), dim=-1)              # (B,1,V)

    expanded_pkv = _repeat_kv_cache(pkv, k)
    prefix_last_lp = prefix_last_lp.repeat_interleave(k, dim=0)   # (Bk,1,V)

    block_input = safe[:, :, P:P + bl].reshape(B * k, bl)
    block_out = ar_model(
        input_ids=block_input,
        past_key_values=expanded_pkv,
        use_cache=False,
    )
    block_lp = F.log_softmax(block_out.logits.float(), dim=-1)    # (Bk,bl,V)

    # prefix_last_lp predicts block[0]; block_lp[:,i] predicts block[i+1]
    all_lp = torch.cat([prefix_last_lp, block_lp[:, :-1]], dim=1) # (Bk,bl,V)
    token_lp = all_lp.gather(
        -1, block_input.unsqueeze(-1)
    ).squeeze(-1)                                                  # (Bk,bl)

    return (
        token_lp * eval_masks.reshape(B * k, bl).float()
    ).sum(-1).view(B, k)


@torch.no_grad()
def _score_block_candidates(
    ar_model,
    x_full: torch.Tensor,
    block_cands: torch.Tensor,
    eval_masks: torch.Tensor,
    block_mask: torch.Tensor,
    block_start: int,
    block_end: int,
) -> torch.Tensor:
    """Score block candidates using the AR model.

    Tries prefix KV-cache first; falls back to a single batched forward
    pass if the cache path fails.

    Args:
        x_full:      ``(B, L)``  current full sequence (may contain masks).
        block_cands: ``(B, k, bl)`` candidate tokens for the block.
        eval_masks:  ``(B, k, bl)`` True/1 at positions to score.
        block_mask:  ``(B, bl)``  True at originally-masked positions.
        block_start, block_end: block boundaries in *x_full*.

    Returns:
        ``(B, k)`` AR log-probability scores (higher is better).
    """
    B, k, bl = block_cands.shape
    device = x_full.device
    ar_vocab = ar_model.config.vocab_size

    # Assemble complete sequences: prefix + candidate block
    base = x_full[:, :block_end].unsqueeze(1).expand(B, k, block_end).clone()
    bmask_exp = block_mask.unsqueeze(1).expand(B, k, bl)
    base[:, :, block_start:block_end][bmask_exp] = block_cands[bmask_exp]

    safe = base.clone()
    safe[safe >= ar_vocab] = 0

    P = block_start

    # ── Fast path: prefix KV-cache ──
    if P >= 1:
        try:
            return _score_with_kv_cache(
                ar_model, safe, eval_masks, P, bl, B, k, device)
        except Exception:
            pass

    # ── Fallback: single batched forward ──
    if P == 0:
        return torch.zeros(B, k, device=device)

    ar_input = safe.reshape(B * k, block_end)
    ar_lp = compute_ar_log_probs(ar_model, ar_input)          # (Bk, L-1)
    eval_full = torch.zeros(B * k, block_end, device=device)
    eval_full[:, P:block_end] = eval_masks.reshape(B * k, bl).float()
    return (ar_lp * eval_full[:, 1:]).sum(-1).view(B, k)


# ────────────────── Invariant energy helpers ──────────────────────────────

def sample_xs_from_xt(xt, x0, mask_id, s_ratio):
    """Sample intermediate xs: keep each masked position masked w.p. s_ratio."""
    is_masked = (xt == mask_id)
    keep = torch.rand_like(xt.float()) < s_ratio
    xs = xt.clone()
    xs[is_masked & ~keep] = x0[is_masked & ~keep]
    return xs


def _diffusion_invariant_correction(
    x_full, x0_candidates, block_mask, block_start,
    mask_id, s_ratio, log_p_x0_given_xt,
):
    B, k, bl = x0_candidates.shape
    corrections = torch.zeros(B, k, device=x_full.device)
    xt_block = x_full[:, block_start:block_start + bl]

    for j in range(k):
        cand = x0_candidates[:, j]
        diff_lp = log_p_x0_given_xt.gather(
            -1, cand.unsqueeze(-1)).squeeze(-1)
        xs = sample_xs_from_xt(xt_block, cand, mask_id, s_ratio)
        is_transition = block_mask & (xs != mask_id)
        is_still = block_mask & (xs == mask_id)
        corrections[:, j] = -(
            (diff_lp * is_transition.float()).sum(-1)
            + (diff_lp * is_still.float()).sum(-1)
        )
    return corrections


# ───────────────── Default reranking (importance sampling) ────────────────

@torch.no_grad()
def energy_rerank(
    x_full: torch.Tensor,
    x0_candidates: torch.Tensor,
    block_mask: torch.Tensor,
    block_start: int,
    block_end: int,
    ar_model,
    mask_id: int,
    ecfg: EnergyConfig,
    log_p_x0_given_xt: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the best full-block candidate (*default* method).

    Returns ``(selected_block, selected_idx)`` — ``(B, bl)`` and ``(B,)``.
    """
    B, k, bl = x0_candidates.shape

    # Evaluate all masked block positions for every candidate
    eval_masks = block_mask.unsqueeze(1).expand(B, k, bl)
    ar_scores = _score_block_candidates(
        ar_model, x_full, x0_candidates, eval_masks, block_mask,
        block_start, block_end)

    if ecfg.energy_type == "invariant" and log_p_x0_given_xt is not None:
        scores = ar_scores + _diffusion_invariant_correction(
            x_full, x0_candidates, block_mask, block_start,
            mask_id, ecfg.s_ratio, log_p_x0_given_xt)
    else:
        scores = ar_scores

    scores = scores - scores.max(dim=-1, keepdim=True)[0]
    weights = F.softmax(scores / ecfg.is_temp, dim=-1)
    idx = torch.multinomial(weights, 1).squeeze(-1)
    arange_B = torch.arange(B, device=x_full.device)
    return x0_candidates[arange_B, idx], idx


# ──────────────── Adaption reranking (trajectory-aware) ───────────────────

@torch.no_grad()
def energy_rerank_adaption(
    x_full: torch.Tensor,
    block_cands: torch.Tensor,
    block_confs: torch.Tensor,
    block_mask: torch.Tensor,
    block_start: int,
    block_end: int,
    ar_model,
    mask_id: int,
    ecfg: EnergyConfig,
    m: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Adaption reranking: per-candidate top-*m* selection + AR scoring.

    For each candidate *x0^j*, the *m* positions with highest DLLM
    confidence are chosen for unmasking, yielding *xs^j*.  The AR model
    scores each *xs^j* at only those *m* transition positions, and the
    candidate with the lowest energy (highest AR score) is selected.

    Args:
        block_cands:  ``(B, k, bl)`` candidate block tokens.
        block_confs:  ``(B, k, bl)`` DLLM confidence (−inf at non-masked).
        block_mask:   ``(B, bl)``    True at masked positions.
        m:            number of positions to unmask.

    Returns:
        ``(selected_block, transfer_mask)`` — ``(B, bl)`` best candidate
        block tokens and ``(B, bl)`` boolean mask of the *m* positions.
    """
    B, k, bl = block_cands.shape
    device = x_full.device

    actual_m = min(m, int(block_mask.sum(dim=-1).min().item()))
    if actual_m <= 0:
        return (
            x_full[:, block_start:block_end].clone(),
            torch.zeros_like(block_mask),
        )

    # Top-m positions per candidate by DLLM confidence
    _, topk_idx = torch.topk(
        block_confs.reshape(B * k, bl), actual_m, dim=-1)
    transfer_masks = torch.zeros(B * k, bl, device=device, dtype=torch.bool)
    transfer_masks.scatter_(1, topk_idx, True)
    transfer_masks = transfer_masks.view(B, k, bl) & block_mask.unsqueeze(1)

    # Score each candidate at its own transition positions
    scores = _score_block_candidates(
        ar_model, x_full, block_cands, transfer_masks, block_mask,
        block_start, block_end)

    # Deterministic selection (lowest energy = highest AR score)
    sel_idx = scores.argmax(dim=-1)
    arange_B = torch.arange(B, device=device)
    return block_cands[arange_B, sel_idx], transfer_masks[arange_B, sel_idx]
