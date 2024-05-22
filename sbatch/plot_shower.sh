#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/net/projects/fermi-1/logs/victor/test_sbatch_plot_shower.stdout
#SBATCH --error=/net/projects/fermi-1/logs/victor/test_sbatch_plot_shower.stderr
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

cd /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code
source /home/${USER}/.bashrc
source activate calo_diffusion
python3 plot_average.py \
    --input_file '/net/projects/fermi-1/data/dataset_1/dataset_1_photons_2.hdf5' \
    --mode 'all' \
    --dataset '1-photons' \
    --output_dir 'shower_plots'\