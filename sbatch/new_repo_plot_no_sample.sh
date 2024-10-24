#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/
#SBATCH --partition=general
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi

python3 scripts/plot.py \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/CaloDiffusionPaper/configs/config_dataset1_photon.json \
    --binning_file /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
    --model_loc /home/${USER}/CaloDiffusionPaper/trained_models/dataset1_photon.pth \
    --model "Diffu" \
    --plot_folder baseline_plots

    