# 1. Load Environment
module load Miniconda3
conda init bash
source ~/.bashrc
module load CUDA/12.1.0
nvcc --version
module unload gcc/13.1.0
gcc --version
conda activate "Your Environment Path"

# 2. Enable full error traces
export HYDRA_FULL_ERROR=1

# 3. Run Perplexity Evaluation
echo "Starting Invariant EBM evaluation at $(date)"
python -u -m main \
  mode=ppl_eval \
  data=openwebtext-split \
  data.cache_dir="Your Dataset Cache Path" \
  model=small \
  model.length=1024 \
  parameterization=subs \
  ebm_backbone=ar \
  use_invariant_ebm=True \
  sampling.intermediate_time_ratio=-1 \
  sampling.partition_mode=analytical \
  sampling.n_partition_samples=4 \
  sampling.use_leave_one_out=True \
  eval.ar_checkpoint_path="Your AR Checkpoint Path" \
  eval.checkpoint_path="Your Invariant EBM Checkpoint Path" \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  T=0 \
  +wandb.offline=true \
  hydra.run.dir=outputs/invariant_ebm_training_eval_random

echo "Invariant EBM evaluation finished at $(date)"