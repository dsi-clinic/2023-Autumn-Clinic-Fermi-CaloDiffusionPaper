#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code
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
    python3 evaluate.py \
	--input_file /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset2_Latent_Diffu_$str_lsu/generated_dataset2_Latent_Diffu.h5 \
	--reference_file /net/projects/fermi-1/data/dataset_2/dataset_2_2.hdf5 \
	--dataset '2' \
    --output_dir evaluation_results/ds2_$str_lsu/

done