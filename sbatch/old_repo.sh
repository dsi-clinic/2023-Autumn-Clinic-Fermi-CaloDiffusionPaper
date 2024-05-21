#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/singh8/old_repo/CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/singh8/old_repo/CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate calo_diffusion
cd /home/${USER}/old_repo/CaloDiffusionPaper/scripts

python3 plot.py \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/old_repo/CaloDiffusionPaper/configs/config_dataset1_photon.json \
    --model_loc /home/${USER}/old_repo/CaloDiffusionPaper/trained_models/dataset1_photon.pth \
    --model "Diffu" \
    --sample \
    --plot_folder baseline_plots
