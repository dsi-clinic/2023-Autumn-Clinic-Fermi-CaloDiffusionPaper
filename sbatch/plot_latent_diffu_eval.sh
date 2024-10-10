#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/net/projects/fermi-1/doug/sbatch_scripts/grey/%j.%N.stdout
#SBATCH --error=/net/projects/fermi-1/doug/sbatch_scripts/grey/%j.%N.stderr
#SBATCH --chdir=/net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --nodelist=k003
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi

python3 scripts/plot.py \
    --data_folder /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/data/dataset_1 \
    --config /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
    --binning_file /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
    --ae_loc /net/projects/fermi-1/doug/ae_models/dataset1_phot_AE/grey_hyper/static_32_32_32_32e240lr0.00004/final.pth \
    --diffu_loc /net/projects/fermi-1/doug/models/dataset1_phot_Latent_Diffu/dataset1_photons_latent_diffu_1/final.pth \
    --model "Latent_Diffu" \
    --layer_sizes 32 32 32 32 \
    --sample \
    --sample_steps 1 \
    --plot_folder /net/projects/fermi-1/doug/plots/latent_plots \
    --save_folder_append "latent_diffu_1"
python3 scripts/plot.py \
    --data_folder /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/data/dataset_1 \
    --config /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
    --binning_file /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
    --ae_loc /net/projects/fermi-1/doug/ae_models/dataset1_phot_AE/grey_hyper/static_32_32_32_32e240lr0.00004/final.pth \
    --diffu_loc /net/projects/fermi-1/doug/models/dataset1_phot_Latent_Diffu/dataset1_photons_latent_diffu_1/final.pth \
    --model "Latent_Diffu" \
    --layer_sizes 32 32 32 32 \
    --plot_folder /net/projects/fermi-1/doug/plots/latent_plots \
    --save_folder_append "latent_diffu_1"

    
