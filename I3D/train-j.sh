#!/bin/bash
#SBATCH --job-name=I3D-RGB
#SBATCH --account=ddt_acc23
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=71:30:00
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail
#SBATCH --mail-type=begin
#SBATCH --mail-user=21010294@st.phenikaa-uni.edu.vn
#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --error=logs/%x_%j_%D.err
#SBATCH --nodelist=hpc24

module purge
module load cuda
module load python

which python

squeue --me
cd /work/21013187/SignLanguageRGBD/all_code/I3D
python train_i3d.py -m rgb -r /work/21013187/SignLanguageRGBD/OutputSplitAbsoluteVer2/ -n 64 -s 10



