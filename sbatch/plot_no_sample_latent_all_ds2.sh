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

layer_size_unet=("16 16 16" "16 16 32 32" "32 32 32" "32 32 32 32")

for lsu in "${layer_size_unet[@]}"
do
    str_lsu="${lsu//" "/"_"}"
    python3 scripts/plot.py \
        --data_folder /net/projects/fermi-1/data/dataset_2 \
        -g /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset2_Latent_Diffu_$str_lsu/generated_dataset2_Latent_Diffu.h5 \
        --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset2.json \
        --binning_file /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_2.xml \
        --plot_folder baseline_plots/ds2_$str_lsu/

done
