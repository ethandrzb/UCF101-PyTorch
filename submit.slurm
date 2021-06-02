#!/bin/bash
#SBATCH --nodes=1                      # Get one node
#SBATCH --cpus-per-task=2              # Two cores per task
#SBATCH --ntasks=2                     # But only one task
#SBATCH --gres=gpu:1                   # And two GPU
#SBATCH --gres-flags=enforce-binding   # Insist on good CPU/GPU alignment
#SBATCH --time=1-00:00:00              # Run for 7 days, at most
#SBATCH --job-name=REU-UCF101              # Name the job so I can see it in squeue
#SBATCH --output=ucf101-%J.out
#SBATCH --mail-type=END,FAIL     # Send me email for various states
#SBATCH --mail-user robert.browning@ucf.edu        # Use this address


# Load modules
module load gcc/gcc-8.3.0
module load anaconda/anaconda3
module load cuda/cuda-10.2

module list                            # Have Nvidia tell us the GPU/CPU mapping so we know

nvidia-smi topo -m

source activate reu2021

python3 --version

CUDA_VISIBLE_DEVICES=0 python3 -u Main.py
