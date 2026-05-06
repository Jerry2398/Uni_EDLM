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

# 3. Run Code
echo "Starting job at $(date)"
python -u -m main \
  mode=train \
  data=openwebtext-split \
  data.cache_dir="Your Dataset Cache Path" \
  model=small \
  model.length=1024 \
  parameterization=subs \
  ebm_backbone=ar \
  use_invariant_ebm=True \
  sampling.intermediate_time_ratio=-1 \
  eval.ar_checkpoint_path="Your AR Checkpoint Path" \
  eval.checkpoint_path=kuleshov-group/mdlm-owt \
  loader.global_batch_size=512 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  trainer.devices=4 \
  trainer.max_steps=1000000 \
  trainer.val_check_interval=10000 \
  trainer.limit_val_batches=0.1 \
  eval.compute_generative_perplexity=True \
  sampling.steps=1000 \
  sampling.num_sample_batches=1 \
  checkpointing.save_dir="Your Invariant EBM Checkpoint Path" \
  checkpointing.resume_from_ckpt=false \
  +wandb.offline=true \
  wandb.name="Your Invariant EBM Wandb Project Name" \
  hydra.run.dir=outputs/invariant_ebm_eval_mode
echo "Job finished at $(date)"