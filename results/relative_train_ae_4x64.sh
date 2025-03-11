#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=logs/%j.%N.stdout
#SBATCH --error=logs/%j.%N.stderr
#SBATCH --chdir=scripts
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Define base directory 
BASE_DIR="$HOME/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper"

source ~/.bashrc
source activate fermi

python3 train_ae.py \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config "$BASE_DIR/configs/config_dataset1_photon.json" \
	--binning_file "$BASE_DIR/CaloChallenge/code/binning_dataset_1_photons.xml" \
	--layer_sizes 64 64 64 64