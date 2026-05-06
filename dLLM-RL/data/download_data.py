import argparse
import os
from huggingface_hub import hf_hub_download
import shutil

DEFAULT_OUTPUT_DIR = "Your Dataset Path"

parser = argparse.ArgumentParser(description="Download a dataset from HF hub")
parser.add_argument(
    "--dataset",
    choices=["PrimeIntellect","MATH_train","demon_openr1math","MATH500","GSM8K","AIME2024","LiveBench","LiveCodeBench","MBPP","HumanEval"],
    required=True,
    help="Which dataset to download"
)
parser.add_argument(
    "--output-dir",
    default=DEFAULT_OUTPUT_DIR,
    help=f"Directory to save dataset JSON files (default: {DEFAULT_OUTPUT_DIR})"
)
args = parser.parse_args()
dataset = args.dataset
output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

if dataset == "MATH_train" or dataset == "PrimeIntellect" or dataset == "demon_openr1math":
    split = "train"
else:
    split = "test"

os.makedirs(output_dir, exist_ok=True)

cached_path = hf_hub_download(
    repo_id=f"Gen-Verse/{dataset}",
    repo_type="dataset",
    filename=f"{split}/{dataset}.json"
)
dest_path = os.path.join(output_dir, f"{dataset}.json")
shutil.copy(cached_path, dest_path)
print(f"Saved to {dest_path}")