#!/bin/bash
#SBATCH --job-name=fermi
#SBATCH --output=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi

python3 CaloChallenge/code/evaluate.py \
    --input_file /net/projects/fermi-1/generated_samples/dataset3/generated.h5 \
    --reference_file /net/projects/fermi-1/data/dataset_3/dataset_3_3.hdf5 \
    --dataset '3' \
    --output_dir /net/projects/fermi-1/evaluation_results/dataset3/ \
    -m all \
    --ratio 