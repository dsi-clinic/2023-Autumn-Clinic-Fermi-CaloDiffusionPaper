#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=logs/%j.%N.stdout
#SBATCH --error=logs/%j.%N.stderr
#SBATCH --chdir=2023-Autumn-Clinic-Fermi-CaloDiffusionPaper
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

BASE_DIR="$HOME/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper"

source ~/.bashrc
source activate fermi

str_lsu="16_16"

python3 train_diffu.py \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config "$BASE_DIR/configs/config_dataset1_photon.json" \
    --binning_file "$BASE_DIR/CaloChallenge/code/binning_dataset_1_photons.xml" \
    --model_loc /net/projects/fermi-1/klin/dataset1_phot_AE_16_16_4e-4/final.pth \
    --layer_sizes 16 16 \
    --model "Latent_Diffu" \
    --load \