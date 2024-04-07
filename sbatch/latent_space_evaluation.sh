#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/net/projects/fermi-1/doug/sbatch_scripts/carina/%j.%N.stdout
#SBATCH --error=/net/projects/fermi-1/doug/sbatch_scripts/carina/%j.%N.stderr
#SBATCH --chdir=chdir=/net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

#Initially Created by Carina. Victor will be modifying or replacing this with his script.
source /home/${USER}/.bashrc
source activate calo_diffusion 
python3 /net/projects/fermi-1/doug/sbatch_scripts/carina/evaluate_latent.py \
    --data_folder /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/data/dataset_2 \
    --config /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset2.json \
    --binning_file /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_2.xml  \   
    --model_loc /net/projects/fermi-1/doug/ae_models/dataset2_AE/downsample_2_8_hrs/final.pth \



