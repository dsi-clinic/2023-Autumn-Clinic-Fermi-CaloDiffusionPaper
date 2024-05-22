#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/net/projects/fermi-1/logs/victor/test_sbatch_grey_logs.stdout
#SBATCH --error=/net/projects/fermi-1/logs/victor/test_sbatch_grey_logs.stderr
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

cd /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper
source /home/${USER}/.bashrc
source activate calo_diffusion
python3 scripts/autoencoder/evaluate_latent.py \
	--data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
    --binning_file /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
	--model_loc /net/projects/fermi-1/grey/ae_models/dataset1_phot_AE/static_32_32_32_32e240lr0.00004/final.pth \
    # --layer_sizes 32_32_32_32