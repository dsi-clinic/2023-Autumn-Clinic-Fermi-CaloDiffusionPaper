#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/aaronz/CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/aaronz/CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/aaronz/CaloDiffusionPaper/scripts
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi

python3 train_diffu.py \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon_test.json \
	--model "Diffu" \
