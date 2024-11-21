#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stdout
#SBATCH --error=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/logs/%j.%N.stderr
#SBATCH --chdir=/home/aaronz/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate fermi

lsu="16 16"
epoch=400
learning_rate=0.00004


str_lsu="${lsu//" "/"_"}"
python3 scripts/plot.py \
    --data_folder /net/projects/fermi-1/data/dataset_1 \
    --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
    --binning_file /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
    --model_loc /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset1_phot_Latent_Diffu_$str_lsu/final.pth \
    --model "Latent_Diffu" \
    --layer_sizes $lsu \
    --encoded_mean_std_loc /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset1_phot_Latent_Diffu_$str_lsu/encoded_mean_std.txt \
    --ae_loc /net/projects/fermi-1/klin/dataset1_phot_AE_16_16_4e-4/final.pth \
    --sample \
    --nevts 121000 \
    --batch_size 1000 \
    --sample_offset 2 \
    --save_folder_append _16_16 \
    --plot_folder baseline_plots/

    