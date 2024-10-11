#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/isaacharlem/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/isaacharlem/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/isaacharlem/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/scripts
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate fermi


python train_diffu.py \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
	--model "Diffu" 