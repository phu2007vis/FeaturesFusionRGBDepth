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
source /home/21010294/VSR/VSREnv/bin/activate
module list
python -c "import sys; print(sys.path)"

which python
python --version
python /home/21010294/VSR/cudacheck.py
squeue --me
cd /home/21010294/ActionRecognition/pytorch-i3d/
python train_i3d.py -m rgb -r /work/21010294/DepthData/OutputSplitAbsoluteVer2/ -n 64 -c "/work/21010294/DepthData/cache/" -s 10



