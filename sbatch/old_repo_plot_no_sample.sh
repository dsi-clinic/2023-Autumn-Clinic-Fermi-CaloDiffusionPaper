#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/aaronz/CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/aaronz/CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi
cd /home/${USER}/CaloDiffusionPaper/scripts

python3 plot.py \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/CaloDiffusionPaper/configs/config_dataset1_photon.json \
    --model_loc /home/${USER}/CaloDiffusionPaper/trained_models/dataset1_photon.pth \
    --model "Diffu" \
    --plot_folder baseline_plots
