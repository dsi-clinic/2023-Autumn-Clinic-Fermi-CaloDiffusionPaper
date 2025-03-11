#!/bin/bash
#SBATCH --job-name=fermi
#SBATCH --output=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=256GB
#SBATCH --time=12:00:00

source /home/${USER}/.bashrc
source activate fermi

python3 scripts/autoencoder/train_ae.py \
    --data_folder /net/projects/fermi-1/data/dataset_3/ \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset3.json \
	--binning_file /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_3.xml \
	--layer_sizes 16 16 16 \
    --save_folder_absolute /net/projects/fermi-1/autoencoders/dataset3/
