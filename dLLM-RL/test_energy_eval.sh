module load Miniconda3
conda init bash
source ~/.bashrc
module load CUDA/12.1.0
nvcc --version
module unload gcc/13.1.0
gcc --version
conda activate "Your Environment Path"
conda list
nvidia-smi

python eval.py config=configs/llada_energy_eval.yaml