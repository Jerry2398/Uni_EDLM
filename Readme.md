# Projects

## 1. Energy-Diffusion-LLM

Official code for DLM experiment of Invariant EBM.

### Setup

```bash
cd Energy-Diffusion-LLM
conda env create -f requirements.yaml
conda activate edlm
```

### Run

```bash
# Training
bash job_inv_edlm_train.sh

# Perplexity evaluation
bash job_inv_edlm_eval.sh

# Sample generation & evaluation
bash job_inv_edlm_sample.sh
```

See `job_inv_edlm_train.sh`, `job_inv_edlm_eval.sh`, and `job_inv_edlm_sample.sh` for full example commands.

---

## 2. dLLM-RL

Official code for DLLM experiment of Invariant EBM.

### Setup

```bash
cd dLLM-RL
conda create -n dllm-rl python=3.10 -y && conda activate dllm-rl
pip install torch==2.6.0
pip install -r requirements.txt
```

### Download Data

```bash
bash download_datasets.sh
```

### Run

```bash
# Evaluation
bash test_energy_eval.sh
```

Swap the YAML file for different models (e.g. `llada_eval.yaml`, `dream_eval.yaml`, `rl_llada.yaml`). Edit the chosen YAML to set your model paths, dataset paths, and GPU count before running.

See `configs/readme.md` for a full list of config files and options.
