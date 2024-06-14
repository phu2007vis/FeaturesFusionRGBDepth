#!/bin/bash
#SBATCH --job-name=extract_pose
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j_%D.out
#SBATCH --account=ddt_acc23
#SBATCH --error=logs/%x_%j_%D.err
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8


module purge
module load python
module list

squeue --me
cd /work/21013187/SignLanguageRGBD
python /work/21013187/SignLanguageRGBD/all_code/resources/utils/mediapipe_extract.py
