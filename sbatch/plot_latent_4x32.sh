#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi

python3 scripts/plot.py \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset1_phot_Latent_Diffu/config_dataset1_photon.json \
    --binning_file /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
    --model_loc /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset1_phot_Latent_Diffu/final.pth \
    --model "Latent_Diffu" \
    --layer_sizes 32 32 32 32 \
    --encoded_mean_std_loc /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset1_phot_Latent_Diffu/encoded_mean_std.txt \
    --ae_loc /net/projects/fermi-1/grey/ae_models/dataset1_phot_AE/static_32_32_32_32e240lr0.00004/final.pth \
    --sample \
    --sample_offset 2 \
    --plot_folder baseline_plots
    