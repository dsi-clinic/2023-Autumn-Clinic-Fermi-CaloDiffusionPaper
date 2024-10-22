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

layer_size_unet=("16 16 16 16 16" "16 16 16 16" "16 16 16 32 32" "32 32 32" "64 64 64")
epoch=400
learning_rate=0.00004

for lsu in "${layer_size_unet[@]}"
do
    str_lsu="${lsu//" "/"_"}"
    python3 scripts/train_diffu.py \
    --data_folder /net/projects/fermi-1/data/dataset_2 \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset2.json \
    --model_loc /net/projects/fermi-1/lijiaxin/ae_models/dataset2_AE/static_$str_lsu"e"$epoch"lr"$learning_rate/final.pth \
    --layer_sizes  $lsu \
    --model "Latent_Diffu" \

done