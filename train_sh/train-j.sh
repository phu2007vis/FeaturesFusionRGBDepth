#!/bin/bash
#SBATCH --job-name=I3D-RGB
#SBATCH --account=ddt_acc23
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=71:30:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err
#SBATCH --nodelist=hpc24

squeue --me
cd /work/21013187/SignLanguageRGBD/all_code
module load python 
python train_sh/train_i3d.py 




