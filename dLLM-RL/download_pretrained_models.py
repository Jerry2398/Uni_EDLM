from huggingface_hub import snapshot_download
import os
target_root = "Your Pretrained Models Path"
# repo_id = "GSAI-ML/LLaDA-8B-Base"  # or GSAI-ML/LLaDA-1.5, etc.
repo_id = "GSAI-ML/LLaDA-8B-Instruct"
local_dir = os.path.join(target_root, repo_id.split("/")[-1])
os.makedirs(target_root, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("Model saved to:", os.path.abspath(local_dir))