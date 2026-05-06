"""LLaDa sampling with Energy-Based Model reranking.

Drop-in replacement for ``llada_sample.py``.  At each denoising step the
script samples *k* x0 candidates from the diffusion model, scores them
with a pretrained AR model, and selects the best candidate before applying
the standard confidence-based unmasking.

Two energy methods are supported via ``energy.energy_method``:
  * ``default``  – rerank full x0 candidates at all masked positions.
  * ``adaption`` – for each candidate select the top-*m* positions by DLLM
    confidence, score only those positions, and use the best partial unmask.
"""
from __future__ import annotations
import math, json, os, time, re, random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from jinja2 import Template
import torch
import torch.nn.functional as F
from termcolor import cprint
from transformers import AutoTokenizer, AutoModel
from llada.modeling_llada import LLaDAModelLM
import multiprocessing as mp

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

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    noise = (- torch.log(noise)) ** temperature
    return logits.exp() / noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num = torch.zeros(mask_num.size(0), steps,
                      device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num[i, :remainder[i]] += 1
    return num


@dataclass
class DiffusionOutput:
    sequences: torch.Tensor
    history:   List[torch.Tensor]
    nfe:       int


# ──────────────────────── token selection (no energy) ──────────────────────

def get_transfer_index(logits, temperature, target, mask_index, x,
                       num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if target == 'confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif target == 'margin_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        top2 = torch.topk(p, 2, dim=-1).values
        x0_p = top2[..., 0] - top2[..., 1]
    elif target == 'neg_entropy':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
    elif target == 'random':
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(target)

    x0 = torch.where(mask_index, x0, x)

    if threshold is not None:
        selected = mask_index & (x0_p >= threshold)
        has_mask = mask_index.any(dim=-1)
        none_sel = (~selected.any(dim=-1)) & has_mask
        if none_sel.any():
            masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
            best_idx = masked_scores.argmax(dim=-1)
            rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
            selected[rows, best_idx[rows]] = True
        return x0, selected

    confidence = x0_p.masked_fill(~mask_index, float("-inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        k = int(num_transfer_tokens[j].item()
                if torch.is_tensor(num_transfer_tokens[j])
                else num_transfer_tokens[j])
        if k <= 0:
            continue
        _, sel = torch.topk(confidence[j], k=k)
        transfer_index[j, sel] = True
    return x0, transfer_index


# ──────────────── energy-aware token selection (default) ──────────────────

def get_transfer_index_energy(
    logits, temperature, target, mask_index, x_window,
    num_transfer_tokens, threshold,
    x_full, block_start, block_end, window_start,
    ar_model, mask_id, ecfg: EnergyConfig,
):
    """Sample *k* candidates and rerank with the AR energy model."""
    B, W, V = logits.shape
    k = ecfg.is_size
    bl = block_end - block_start
    block_offset = block_start - window_start

    # 1. Sample k candidates (window scope)
    candidates_win = []
    for _ in range(k):
        noisy = add_gumbel_noise(logits, temperature=temperature)
        x0_k = torch.argmax(noisy, dim=-1)
        x0_k = torch.where(mask_index, x0_k, x_window)
        candidates_win.append(x0_k)
    x0_stack_win = torch.stack(candidates_win, dim=1)          # (B, k, W)

    # 2. Extract block portion & rerank
    block_cands = x0_stack_win[:, :, block_offset:block_offset + bl]
    block_mask  = mask_index[:, block_offset:block_offset + bl]

    log_p = None
    if ecfg.energy_type == "invariant":
        log_p = F.log_softmax(
            logits[:, block_offset:block_offset + bl].float(), dim=-1)

    best_block, sel_idx = energy_rerank(
        x_full, block_cands, block_mask,
        block_start, block_end,
        ar_model, mask_id, ecfg, log_p)

    # Rebuild full-window x0 from selected candidate
    x0 = x0_stack_win[torch.arange(B, device=logits.device), sel_idx]

    # 3. Standard confidence & unmasking
    if target == 'confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif target == 'margin_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        top2 = torch.topk(p, 2, dim=-1).values
        x0_p = top2[..., 0] - top2[..., 1]
    elif target == 'neg_entropy':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = -torch.sum(p * torch.log(p + 1e-10), dim=-1)
    elif target == 'random':
        x0_p = torch.rand(x0.shape, device=x0.device)
    else:
        raise NotImplementedError(target)

    x0 = torch.where(mask_index, x0, x_window)

    if threshold is not None:
        selected = mask_index & (x0_p >= threshold)
        has_mask = mask_index.any(dim=-1)
        none_sel = (~selected.any(dim=-1)) & has_mask
        if none_sel.any():
            masked_scores = x0_p.masked_fill(~mask_index, float("-inf"))
            best_idx = masked_scores.argmax(dim=-1)
            rows = torch.nonzero(none_sel, as_tuple=False).squeeze(-1)
            selected[rows, best_idx[rows]] = True
        return x0, selected

    confidence = x0_p.masked_fill(~mask_index, float("-inf"))
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        kk = int(num_transfer_tokens[j].item()
                 if torch.is_tensor(num_transfer_tokens[j])
                 else num_transfer_tokens[j])
        if kk <= 0:
            continue
        _, sel = torch.topk(confidence[j], k=kk)
        transfer_index[j, sel] = True
    return x0, transfer_index


# ──────────────── energy-aware token selection (adaption) ─────────────────

def get_transfer_index_energy_adaption(
    logits, temperature, mask_index, x_window,
    num_transfer_tokens,
    x_full, block_start, block_end, window_start,
    ar_model, mask_id, ecfg: EnergyConfig,
):
    """Adaption method: sample k candidates, pick top-m positions per
    candidate by DLLM confidence, score the partial unmasks with the AR
    model, and return the best one as the decoding result.
    """
    B, W, V = logits.shape
    k = ecfg.is_size
    bl = block_end - block_start
    block_offset = block_start - window_start

    # Determine m (unmask count)
    m = ecfg.adaption_m
    if m <= 0:
        if torch.is_tensor(num_transfer_tokens):
            nt = num_transfer_tokens.flatten()
            m = max(1, int(nt.max().item()))
        else:
            m = max(1, int(num_transfer_tokens))

    block_logits = logits[:, block_offset:block_offset + bl]     # (B, bl, V)
    block_mask = mask_index[:, block_offset:block_offset + bl]   # (B, bl)
    p = F.softmax(block_logits.float(), dim=-1)                  # (B, bl, V)

    # Sample k candidates for the full window
    candidates_win = []
    for _ in range(k):
        noisy = add_gumbel_noise(logits, temperature=temperature)
        x0_k = torch.argmax(noisy, dim=-1)
        x0_k = torch.where(mask_index, x0_k, x_window)
        candidates_win.append(x0_k)
    x0_stack_win = torch.stack(candidates_win, dim=1)            # (B, k, W)
    block_cands = x0_stack_win[:, :, block_offset:block_offset + bl]

    # Per-candidate confidence: p(chosen_token | context) at each position
    block_confs = torch.stack([
        p.gather(-1, block_cands[:, j].unsqueeze(-1)).squeeze(-1)
        for j in range(k)
    ], dim=1)                                                    # (B, k, bl)
    block_confs = block_confs.masked_fill(
        ~block_mask.unsqueeze(1), float('-inf'))

    selected_block, transfer_mask_block = energy_rerank_adaption(
        x_full, block_cands, block_confs, block_mask,
        block_start, block_end, ar_model, mask_id, ecfg, m)

    # Build window-level returns
    x0 = x_window.clone()
    x0[:, block_offset:block_offset + bl] = torch.where(
        block_mask, selected_block,
        x_window[:, block_offset:block_offset + bl])

    transfer_index = torch.zeros(B, W, device=logits.device, dtype=torch.bool)
    transfer_index[:, block_offset:block_offset + bl] = transfer_mask_block
    return x0, transfer_index


# ──────────────────────────── generation loop ──────────────────────────────

@torch.no_grad()
def generate_with_energy_cache(
    model, prompt,
    steps, gen_length, block_length, temperature,
    target, mask_id, further_horizon, use_cache, unmask_threshold,
    ar_model, ecfg: EnergyConfig,
) -> DiffusionOutput:

    cgws = further_horizon
    B, L0 = prompt.shape
    x = torch.full((B, L0 + gen_length), mask_id,
                   dtype=torch.long, device=prompt.device)
    max_length = L0 + gen_length
    x[:, :L0] = prompt

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    base_s, rem = divmod(steps, num_blocks)
    steps_per_block = [base_s + (i < rem) for i in range(num_blocks)]

    nfe = 0
    energy_step_counter = 0
    # With dynamic threshold the loop runs until the block is fully unmasked,
    # so actual total iterations ≈ gen_length, not `steps`.  Use the larger
    # of the two to correctly honour energy_fraction as "first X% of *actual*
    # denoising steps."
    estimated_total_iters = max(steps, gen_length) if unmask_threshold is not None else steps
    energy_cutoff = int(estimated_total_iters * ecfg.energy_fraction)
    hist: List[torch.Tensor] = []
    is_adaption = ecfg.energy_method == "adaption"

    for blk in range(num_blocks):
        s = L0 + blk * block_length
        e = L0 + (blk + 1) * block_length

        if cgws is not None:
            window_end   = min(e + cgws, max_length)
            window_slice = slice(s, window_end)

        cur_steps = steps_per_block[blk]
        num_transfer = get_num_transfer_tokens(
            (x[:, s:e] == mask_id), cur_steps)

        # ── first forward (full sequence for prefix KV) ──
        if use_cache:
            out = model(x, use_cache=True)
            pkv = tuple(tuple(t[:, :, :s] for t in layer)
                        for layer in out.past_key_values)
        else:
            out = model(x, use_cache=False)

        mask_all = (x == mask_id)
        mask_all[:, e:] = 0

        use_energy = (ecfg.energy_frequency > 0
                      and energy_step_counter < energy_cutoff
                      and energy_step_counter % ecfg.energy_frequency == 0)
        energy_step_counter += 1

        if use_energy:
            if is_adaption:
                x0, tr_idx = get_transfer_index_energy_adaption(
                    out.logits, temperature, mask_all, x,
                    num_transfer[:, 0],
                    x, s, e, 0, ar_model, mask_id, ecfg)
            else:
                x0, tr_idx = get_transfer_index_energy(
                    out.logits, temperature, target, mask_all, x,
                    num_transfer[:, 0], unmask_threshold,
                    x, s, e, 0, ar_model, mask_id, ecfg)
        else:
            x0, tr_idx = get_transfer_index(
                out.logits, temperature, target, mask_all, x,
                num_transfer[:, 0], unmask_threshold)

        x[tr_idx] = x0[tr_idx]
        hist.append(x.clone().cpu())
        nfe += 1

        # ── subsequent steps within block ──
        i = 1
        while True:
            nfe += 1
            if cgws is not None:
                mask_blk = (x[:, window_slice] == mask_id)
            else:
                mask_blk = (x[:, s:] == mask_id)
            mask_blk[:, block_length:] = 0

            if use_cache:
                if cgws is not None:
                    logits = model(x[:, window_slice],
                                   past_key_values=pkv,
                                   use_cache=True).logits
                else:
                    logits = model(x[:, s:],
                                   past_key_values=pkv,
                                   use_cache=True).logits
            else:
                logits = model(x, use_cache=False).logits
                logits = logits[:, s:]

            use_energy = (ecfg.energy_frequency > 0
                          and energy_step_counter < energy_cutoff
                          and energy_step_counter % ecfg.energy_frequency == 0)
            energy_step_counter += 1

            x_window = x[:, window_slice] if cgws is not None else x[:, s:]

            # Safely clamp step index for num_transfer lookup
            step_i = min(i, num_transfer.shape[1] - 1)

            if use_energy:
                if is_adaption:
                    x0, tr_idx = get_transfer_index_energy_adaption(
                        logits, temperature, mask_blk, x_window,
                        num_transfer[:, step_i],
                        x, s, e, s, ar_model, mask_id, ecfg)
                else:
                    x0, tr_idx = get_transfer_index_energy(
                        logits, temperature, target, mask_blk, x_window,
                        num_transfer[:, step_i], unmask_threshold,
                        x, s, e, s, ar_model, mask_id, ecfg)
            else:
                x0, tr_idx = get_transfer_index(
                    logits, temperature, target, mask_blk, x_window,
                    num_transfer[:, step_i], unmask_threshold)

            if cgws is not None:
                x[:, window_slice][tr_idx] = x0[tr_idx]
            else:
                x[:, s:][tr_idx] = x0[tr_idx]

            hist.append(x.clone().cpu())

            if (x[:, s:e] == mask_id).sum() == 0:
                break
            i += 1

    return DiffusionOutput(sequences=x, history=hist, nfe=nfe)


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


def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    return matches[-1].strip() if matches else "We can not extract the code in the output. "


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

    model_gpu = (LLaDAModelLM
                 .from_pretrained(pretrained_model,
                                  trust_remote_code=True,
                                  torch_dtype=torch.bfloat16)
                 .to(device).eval())

    ecfg = EnergyConfig(**ecfg_dict)
    ar_model = load_ar_model(
        ar_model_path, device,
        load_in_4bit=ecfg.ar_load_in_4bit,
        load_in_8bit=ecfg.ar_load_in_8bit)

    tokenizer_gpu = AutoTokenizer.from_pretrained(
        pretrained_model, trust_remote_code=True)

    for start in tqdm(range(0, len(prompts), batch_size),
                      desc=f"GPU {rank}", position=rank, leave=True):
        batch_prompts = prompts[start:start + batch_size]
        batch_idxs    = orig_idx[start:start + batch_size]

        enc = tokenizer_gpu(batch_prompts, padding=True,
                            return_tensors="pt", padding_side="left")
        input_ids = enc["input_ids"].to(device)
        mask_id = tokenizer_gpu.encode('<|mdm_mask|>')[0]

        fh = config.rollout.further_horizon
        if not config.rollout.use_cache:
            fh = None

        unmask_threshold = (None
                            if config.rollout.remasking_strategy == "low_confidence_static"
                            else config.rollout.dynamic_threshold)

        out = generate_with_energy_cache(
            model_gpu, input_ids,
            steps=config.rollout.steps,
            gen_length=config.rollout.max_gen_length,
            block_length=config.rollout.block_size,
            temperature=config.rollout.temperature,
            target=config.rollout.target,
            mask_id=mask_id,
            further_horizon=fh,
            use_cache=config.rollout.use_cache,
            unmask_threshold=unmask_threshold,
            ar_model=ar_model,
            ecfg=ecfg,
        )
        out.sequences = out.sequences.cpu()
        torch.cuda.empty_cache()

        seq_ids = out.sequences[:, input_ids.shape[1]:].tolist()
        texts = tokenizer_gpu.batch_decode(
            seq_ids, skip_special_tokens=False,
            clean_up_tokenization_spaces=True)

        for i, idx in enumerate(batch_idxs):
            m = denoise_step_map(out.history, mask_id=mask_id, sample_idx=i)
            step_map = m[input_ids.shape[1]:].tolist()
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
        """<|startoftext|><|start_header_id|>user<|end_header_id|>"""
        """You need to put your final answer in \\boxed{}. """
        """This is the problem:\n{{problem}}"""
        """<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n"""
    )

    code_eval = False

    dataset = config.dataset.eval_dataset
    pretrained_model = config.model
    if config.dataset.data_type == "code":
        code_eval = True
        system_prompts_function = (
            '''<|startoftext|><|start_header_id|>user<|end_header_id|>'''
            '''{{problem}}\nPlace your code within a single Python code block '''
            '''```python ```. Do not include more than one code block. '''
            '''<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n''')
        system_prompts_stdio = (
            '''<|startoftext|><|start_header_id|>user<|end_header_id|>'''
            '''This is the problem:\n{{problem}}\n '''
            '''You should put your code in ```python ```. '''
            '''Use input() to read input and print() to produce output in your script. '''
            '''<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n''')
    elif config.dataset.data_type == "option":
        system_prompts = (
            '''<|startoftext|><|start_header_id|>user<|end_header_id|>'''
            '''This is the problem:\n{{problem}}\n'''
            '''You need to think step by step and put the final option '''
            '''(A, B, C, or D only—no other character) in \\boxed{}. '''
            '''<|eot_id|><|startoftext|><|start_header_id|>assistant<|end_header_id|>\n''')

    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    data_dir = OmegaConf.select(config, "dataset.data_dir", default="../data")
    with open(os.path.join(data_dir, dataset + ".json"), 'r') as f:
        data = json.load(f)

    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    if num_node > 1:
        data = get_data_chunk(data, num_node, node_index)

    num = len(data)
    tokenizer = AutoTokenizer.from_pretrained(
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

    cprint("start generation (energy-enhanced LLaDa)...", "green")
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
