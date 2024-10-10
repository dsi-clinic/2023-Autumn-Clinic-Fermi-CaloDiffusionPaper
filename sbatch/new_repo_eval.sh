#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi

python3 evaluate.py \
	--input_file /home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset1_phot_Diffu/generated_dataset1_phot_Diffu.h5 \
	--reference_file /net/projects/fermi-1/data/dataset_1/dataset_1_photons_1.hdf5 \
	--dataset '1-photons'
