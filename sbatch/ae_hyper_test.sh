#!/bin/bash
source /home/${USER}/.bashrc
#SBa

# Note that this script submits multiple jobs using sbatch.

# Set the conda environment to "fermi"
env="fermi"

# Dataset selection (adjust as needed)
dataset="2train"  # Options: "1", "2train", "2eval"

# Hyperparameter arrays
layer_size_unet=( "16 16 16" "16 16 32" "32 32 32" )
learning_rate=( 0.0004 0.0001 0.00004 )

# Path to your base job script (update the path accordingly)
job_script="/home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/sbatch/autoscript_ae_ds${dataset}.sh"

# Update the job script to activate the "fermi" conda environment
sed -i "s|^source activate .*|source activate ${env}|" "${job_script}"

# Iterate over hyperparameter combinations
for lsu in "${layer_size_unet[@]}"
do
    # Update the layer_size_unet in the job script
    sed -i "s|^layer_size_unet=.*|layer_size_unet=( ${lsu} )|" "${job_script}"
    for lr in "${learning_rate[@]}"
    do
        # Update the learning_rate in the job script
        sed -i "s|^learning_rate=.*|learning_rate=${lr}|" "${job_script}"
        
        # Optionally, set a unique job name incorporating hyperparameters
        job_name="ae_ds${dataset}_lsu${lsu// /_}_lr${lr}"
        sed -i "s|^#SBATCH --job-name=.*|#SBATCH --job-name=${job_name}|" "${job_script}"
        
        # Submit the job
        sbatch_output=$(sbatch "${job_script}")
        echo "${sbatch_output}"
    done
done
