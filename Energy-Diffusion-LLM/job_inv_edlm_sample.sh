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

echo "Starting Invariant EBM sample evaluation at $(date)"
python -u -m main \
  mode=sample_eval \
  data=openwebtext-split \
  data.cache_dir="Your Dataset Cache Path" \
  model=small \
  model.length=1024 \
  parameterization=subs \
  ebm_backbone=ar \
  use_invariant_ebm=True \
  sampling.intermediate_time_ratio=-1 \
  eval.ar_checkpoint_path="Your AR Checkpoint Path" \
  eval.checkpoint_path="Your Invariant EBM Checkpoint Path" \
  eval.gen_ppl_eval_model_name_or_path=gpt2-large \
  sampling.predictor=ddpm_cache \
  sampling.steps=1024 \
  sampling.is_size=2 \
  sampling.is_start=1.0 \
  sampling.is_end=0.0 \
  sampling.is_temp=1 \
  loader.eval_batch_size=1 \
  sampling.num_sample_batches=128 \
  +wandb.offline=true \
  hydra.run.dir=outputs/invariant_ebm_sample_eval_pretrained

echo "Invariant EBM sample evaluation (pretrained) finished at $(date)"
