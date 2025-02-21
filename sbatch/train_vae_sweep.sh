#!/bin/bash
#
#SBATCH --job-name=vae_sweep
#SBATCH --output=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/scripts
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-11

source /home/${USER}/.bashrc
source activate fermi

# Define hyperparameter combinations
declare -a betas=(0.1 0.5 1.0 2.0)
declare -a warmup_steps=(100 500 1000)
declare -a loss_types=("mse" "huber")
declare -a zdims=(16 32 64)
declare -a hidden_sizes=(32 64 128)

total_combinations=$((${#betas[@]} * ${#warmup_steps[@]} * ${#loss_types[@]}))
beta_idx=$(( SLURM_ARRAY_TASK_ID % ${#betas[@]} ))
warmup_idx=$(( (SLURM_ARRAY_TASK_ID / ${#betas[@]}) % ${#warmup_steps[@]} ))
loss_idx=$(( SLURM_ARRAY_TASK_ID / (${#betas[@]} * ${#warmup_steps[@]}) ))

beta=${betas[$beta_idx]}
warmup=${warmup_steps[$warmup_idx]}
loss_type=${loss_types[$loss_idx]}

FOLDER_NAME="vae_beta${beta}_warmup${warmup}_${loss_type}"

python3 autoencoder/train_ae.py \
    --model VAE \
    --vae_zdim 32 \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
    --binning_file /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
    --save_folder_absolute "/net/projects/fermi-1/autoencoders/dataset1/sweeps/${FOLDER_NAME}" \
    --layer_sizes 32 32 32 32 \
    --learning_rate 1e-4 \
    --beta $beta \
    --kl_warmup_steps $warmup \
    --loss_type $loss_type 