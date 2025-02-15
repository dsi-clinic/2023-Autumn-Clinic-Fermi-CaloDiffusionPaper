#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi

python3 scripts/plot.py \
    --data_folder /net/projects/fermi-1/data/dataset_3 \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset3_Latent_Diffu/config_dataset3.json \
    --model_loc /net/projects/fermi-1/diffusors/dataset3/best_val.pth \
    --model "Latent_Diffu" \
    --layer_sizes 32 32 32 32 \
    --encoded_mean_std_loc /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset3_Latent_Diffu/encoded_mean_std.txt \
    --ae_loc /net/projects/fermi-1/autoencoders/dataset3/dataset3_AE_32_32_32_32_0.0001/best_val.pth \
    --sample \
    --sample_offset 88 \
    --plot_folder /net/projects/fermi-1/dataset3/plots \
    --generated /net/projects/fermi-1/generated_samples/dataset3/generated.h5

