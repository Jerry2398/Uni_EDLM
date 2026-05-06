#!/usr/bin/env python3
"""Download Qwen AR models from Hugging Face to a local directory.

Use these for energy-based reranking (ar_model_path in llada_energy_eval / dream_energy_eval).
Run from project root:

  python download_qwen_ar_models.py --output-dir "Your Pretrained Models Path"
  python download_qwen_ar_models.py --output-dir "Your Pretrained Models Path" --model Qwen2-7B --model Qwen2.5-7B

If --model is not given, downloads all supported Qwen AR models used in energy configs.
"""

import argparse
import os
from huggingface_hub import snapshot_download

DEFAULT_OUTPUT_DIR = "Your Pretrained Models Path"

# repo_id -> local folder name (last part of repo_id)
SUPPORTED_QWEN_AR_MODELS = [
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-4B",
]


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen AR models for energy-based evaluation."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save models (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=["Qwen2-7B", "Qwen2.5-7B", "Qwen3-8B", "Qwen3-4B"],
        help="Model to download (can be repeated). If not set, download all supported.",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    if args.models:
        repo_ids = [f"Qwen/{m}" for m in args.models]
    else:
        repo_ids = SUPPORTED_QWEN_AR_MODELS

    for repo_id in repo_ids:
        local_name = repo_id.split("/")[-1]
        local_dir = os.path.join(output_dir, local_name)
        print(f"Downloading {repo_id} -> {local_dir} ...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"  -> {os.path.abspath(local_dir)}")

    print("\nDone. Set energy.ar_model_path in your config to the full path, e.g.:")
    print(f'  ar_model_path: "{output_dir}/Qwen2-7B"')
    print(f'  ar_model_path: "{output_dir}/Qwen2.5-7B"')


if __name__ == "__main__":
    main()
