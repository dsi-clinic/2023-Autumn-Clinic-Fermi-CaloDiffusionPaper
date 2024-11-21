#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi

str_lsu="16_16"

python3 scripts/train_diffu.py \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
    --binning_file /home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
    --model_loc /net/projects/fermi-1/klin/dataset1_phot_AE_16_16_4e-4/final.pth \
    --layer_sizes 16 16 \
    --model "Latent_Diffu" \
    --load \