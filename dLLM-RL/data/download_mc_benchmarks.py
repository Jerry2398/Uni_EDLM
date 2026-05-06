#!/usr/bin/env python3
"""Download multiple-choice benchmarks (MMLU, MMLU-Pro, HellaSwag, ARC-C, GPQA)
from HuggingFace and save as JSON in dLLM-RL format.

Output format per item: {"question": "...", "ground_truth_answer": "A"}

Usage (from project root or from data/):
  python data/download_mc_benchmarks.py --output-dir "Your Dataset Path"
  python data/download_mc_benchmarks.py --output-dir "Your Dataset Path" --benchmark MMLU --benchmark HellaSwag

Requires: pip install datasets
"""

import argparse
import json
import os

DEFAULT_OUTPUT_DIR = "Your Dataset Path"
LETTERS_4 = ["A", "B", "C", "D"]
LETTERS_10 = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def _ensure_datasets():
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError("Please install: pip install datasets")


def build_mmlu(output_dir: str) -> None:
    """MMLU (cais/mmlu): 57 subjects, test split, 4 options."""
    ds = _ensure_datasets()
    from datasets import get_dataset_config_names, load_dataset, concatenate_datasets

    configs = get_dataset_config_names("cais/mmlu")
    all_test = []
    for cfg in configs:
        try:
            d = load_dataset("cais/mmlu", cfg, split="test")
            all_test.append(d)
        except Exception as e:
            print(f"  Skip config {cfg}: {e}")
    if not all_test:
        raise RuntimeError("MMLU: no configs loaded")
    combined = concatenate_datasets(all_test)
    out = []
    for row in combined:
        q = row["question"]
        choices = row["choices"]
        for i, c in enumerate(choices):
            q += f"\n{LETTERS_4[i]}. {c}"
        ans = row["answer"]
        if isinstance(ans, int) and 0 <= ans < 4:
            gt = LETTERS_4[ans]
        else:
            gt = str(ans).strip().upper()
            if gt in ("0", "1", "2", "3"):
                gt = LETTERS_4[int(gt)]
            elif gt not in LETTERS_4 and len(gt) == 1 and ord(gt) >= ord("A") and ord(gt) <= ord("D"):
                pass
        out.append({"question": q, "ground_truth_answer": gt})
    path = os.path.join(output_dir, "MMLU.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"MMLU: {len(out)} items -> {path}")


def build_mmlu_pro(output_dir: str) -> None:
    """MMLU-Pro (TIGER-Lab/MMLU-Pro): 10 options A–J. Columns: question, options, answer."""
    ds = _ensure_datasets()
    from datasets import load_dataset

    try:
        d = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    except Exception:
        d = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    out = []
    for row in d:
        q = row.get("question", row.get("Question", ""))
        opts = row.get("options", row.get("choices", row.get("Options", [])))
        if isinstance(opts, str):
            import json as _json
            opts = _json.loads(opts) if opts.startswith("[") else []
        for i, c in enumerate(opts):
            letter = LETTERS_10[i] if i < len(LETTERS_10) else str(i + 1)
            q += f"\n{letter}. {c}"
        ans = row.get("answer", row.get("Answer", row.get("answer_key", row.get("answer_index", "A"))))
        if isinstance(ans, int):
            gt = LETTERS_10[ans] if ans < len(LETTERS_10) else str(ans)
        else:
            gt = str(ans).strip().upper()
            if len(gt) == 1 and ord(gt) >= ord("A") and ord(gt) <= ord("J"):
                pass
            elif gt.isdigit() and 0 <= int(gt) < 10:
                gt = LETTERS_10[int(gt)]
        out.append({"question": q, "ground_truth_answer": gt})
    path = os.path.join(output_dir, "MMLUPro.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"MMLU-Pro: {len(out)} items -> {path}")


def build_hellaswag(output_dir: str) -> None:
    """HellaSwag (Rowan/hellaswag): context + 4 endings."""
    ds = _ensure_datasets()
    from datasets import load_dataset

    d = load_dataset("Rowan/hellaswag", split="test")
    out = []
    for row in d:
        ctx = row["ctx"]
        endings = row["endings"]
        if isinstance(endings, str):
            import json as _json
            endings = _json.loads(endings)
        q = "Context: " + ctx + "\n\nWhich ending is most plausible?\n"
        for i, e in enumerate(endings[:4]):
            q += f"{LETTERS_4[i]}. {e}\n"
        label = row["label"]
        try:
            idx = int(label) if isinstance(label, str) else int(label)
        except (ValueError, TypeError):
            idx = 0
        idx = max(0, min(idx, 3))
        out.append({"question": q.strip(), "ground_truth_answer": LETTERS_4[idx]})
    path = os.path.join(output_dir, "HellaSwag.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"HellaSwag: {len(out)} items -> {path}")


def build_arc_c(output_dir: str) -> None:
    """ARC-Challenge (allenai/ai2_arc)."""
    ds = _ensure_datasets()
    from datasets import load_dataset

    d = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    out = []
    for row in d:
        q = row["question"]
        choices = row["choices"]
        labels = choices.get("label", ["A", "B", "C", "D"])
        texts = choices.get("text", [])
        for lb, tx in zip(labels, texts):
            q += f"\n{lb}. {tx}"
        key = row.get("answerKey", row.get("answer_key", "A"))
        gt = str(key).strip().upper()
        if len(gt) == 1 and gt not in labels and ord(gt) >= ord("A") and ord(gt) <= ord("D"):
            pass
        elif gt.isdigit() and 1 <= int(gt) <= 4:
            gt = LETTERS_4[int(gt) - 1]
        out.append({"question": q, "ground_truth_answer": gt})
    path = os.path.join(output_dir, "ARC_C.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"ARC-C: {len(out)} items -> {path}")


def build_gpqa(output_dir: str) -> None:
    """GPQA Diamond (aradhye/gpqa_diamond): 198 questions, 4 options."""
    ds = _ensure_datasets()
    from datasets import load_dataset

    d = load_dataset("aradhye/gpqa_diamond", split="train")
    # Column names may vary
    out = []
    for row in d:
        q = row.get("question", row.get("question_text", row.get("text", "")))
        if not q and "problem" in row:
            q = row["problem"]
        # Some GPQA versions have options in one string or list
        opts = row.get("options", row.get("choices", []))
        if isinstance(opts, str):
            import json as _json
            try:
                opts = _json.loads(opts)
            except Exception:
                opts = []
        if opts:
            for i, c in enumerate(opts[:4]):
                q += f"\n{LETTERS_4[i]}. {c}"
        ans = row.get("answer", row.get("correct_answer", "A"))
        gt = str(ans).strip().upper()
        if len(gt) != 1 or gt not in LETTERS_4:
            if gt.isdigit() and 0 <= int(gt) < 4:
                gt = LETTERS_4[int(gt)]
            else:
                gt = "A"
        out.append({"question": q, "ground_truth_answer": gt})
    path = os.path.join(output_dir, "GPQA.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"GPQA: {len(out)} items -> {path}")


BUILDERS = {
    "MMLU": build_mmlu,
    "MMLUPro": build_mmlu_pro,
    "HellaSwag": build_hellaswag,
    "ARC_C": build_arc_c,
    "GPQA": build_gpqa,
}


def main():
    parser = argparse.ArgumentParser(description="Download MC benchmarks to dLLM-RL JSON format")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        choices=list(BUILDERS.keys()),
        help="Benchmark(s) to download (default: all)",
    )
    args = parser.parse_args()
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    benchmarks = args.benchmarks or list(BUILDERS.keys())
    for name in benchmarks:
        try:
            BUILDERS[name](output_dir)
        except Exception as e:
            print(f"Failed {name}: {e}")
            raise
    print("\nDone. In config use dataset.eval_dataset and dataset.data_type:")
    print("  eval_dataset: MMLU | MMLUPro | HellaSwag | ARC_C | GPQA")
    print("  data_type: option")


if __name__ == "__main__":
    main()
