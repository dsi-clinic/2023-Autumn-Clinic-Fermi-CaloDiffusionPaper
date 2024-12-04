#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/scripts
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi

python3 train_diffu.py \
    --data_folder /net/projects/fermi-1/data/dataset_3/ \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset3.json \
    --model_loc /home/simonkatz/ae_models/dataset3_AE_32_32_32_32_0.0001/final.pth \
    --layer_sizes 16 16 16 16 \
    --model "Latent_Diffu" \