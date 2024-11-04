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

layer_size_unet=("16 16 16 16 16" "16 16 16 16" "16 16 16 32 32")
batch_sizes=("10" "100" "1000")
epoch=400
learning_rate=0.00004

for lsu in "${layer_size_unet[@]}"
do
    str_lsu="${lsu//" "/"_"}"
    for batch in "${batch_sizes[@]}"
    do 
        events=$(( 10 * $batch ))
        python3 scripts/plot.py \
        --data_folder /net/projects/fermi-1/data/dataset_2 \
        --config /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset2.json \
        --binning_file /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_2.xml \
        --model_loc /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset2_Latent_Diffu_$str_lsu/final.pth \
        --model "Latent_Diffu" \
        --layer_sizes $lsu \
        --encoded_mean_std_loc /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/models/dataset2_Latent_Diffu_$str_lsu/encoded_mean_std.txt \
        --ae_loc /net/projects/fermi-1/lijiaxin/ae_models/dataset2_AE/static_$str_lsu"e"$epoch"lr"$learning_rate/final.pth \
        --sample \
        --nevts $events \
        --batch_size $batch \
        --sample_offset 2 \
        --plot_folder baseline_plots/
    done

done
    