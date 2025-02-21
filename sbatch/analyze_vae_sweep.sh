#!/bin/bash
#
#SBATCH --job-name=analyze_vae_sweep
#SBATCH --output=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/simonkatz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper
#SBATCH --partition=general
#SBATCH --time=1:00:00

source /home/${USER}/.bashrc
source activate fermi

python scripts/analyze_vae_sweep.py \
  --sweep_folder /net/projects/fermi-1/autoencoders/dataset1/sweeps/ \