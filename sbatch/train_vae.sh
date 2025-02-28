#!/bin/bash
#
#SBATCH --job-name=fermi_vae
#SBATCH --output=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/scripts
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

source /home/${USER}/.bashrc
source activate fermi

python autoencoder/train_ae.py \
    --model VAE \
    --vae_zdim 32 \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
    --binning_file /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
    --save_folder_absolute /net/projects/fermi-1/autoencoders/dataset1/ \
    --layer_sizes 32 32 32 32 \
    --learning_rate 1e-4 \
    --nevts 128 \
    --frac 1.0 \
    --no_early_stop \
    --max_epochs 500