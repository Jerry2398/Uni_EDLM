# 1. Load Environment
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

# 2. Move to working directory (PBS starts in home by default)
# cd $PBS_O_WORKDIR

# 3. Run Code
# python eval.py config=configs/trado_eval.yaml
python eval.py config=configs/llada_energy_eval.yaml