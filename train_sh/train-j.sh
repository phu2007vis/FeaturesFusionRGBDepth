#!/bin/bash
#SBATCH --job-name=I3D-RGB
#SBATCH --partition=dgx-small
#SBATCH --time=69:00:00
#SBATCH --account=ddt_acc23
#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err

squeue --me
cd /work/21013187/SignLanguageRGBD/all_code
module load python 
module load cuda
nvidia-smi
python /work/21013187/SignLanguageRGBD/all_code/train_sh/train.py --device="cuda:2"


