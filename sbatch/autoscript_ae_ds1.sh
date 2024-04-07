#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/net/projects/fermi-1/doug/sbatch_scripts/grey/%j.%N.stdout
#SBATCH --error=/net/projects/fermi-1/doug/sbatch_scripts/grey/%j.%N.stderr
#SBATCH --chdir=/net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
source /home/${USER}/.bashrc
source activate calo_diffusion

layer_size_unet=( "16 16 16 16" )
epoch_start_eval=20 #default 20
epochs_per_eval=20 #default 20
epoch_max=400 #default 400
learning_rate=0.00004

for lsu in "${layer_size_unet[@]}"
do
    str_lsu="${lsu//" "/"_"}"
    for epoch in $(seq $epoch_start_eval $epochs_per_eval $epoch_max)
    do
        if [ $epoch -ne $epoch_start_eval ]; then
            prev_epoch=$((epoch-epochs_per_eval))
            cp -r "/net/projects/fermi-stimulations/ae_models/dataset1_phot_AE/grey_hyper/static_$str_lsu"e"$prev_epoch"lr"$learning_rate" \
                "/net/projects/fermi-stimulations/ae_models/dataset1_phot_AE/grey_hyper/static_$str_lsu"e"$epoch"lr"$learning_rate"
        else
            echo "layer_size_unet=$lsu" > "/net/projects/fermi-stimulations/ae_models/dataset1_phot_AE/grey_hyper/static_$str_lsu"e"$epoch"lr"$learning_rate/training_time.txt"
        fi

        start=`date +%s`
        python3 scripts/autoencoder/grey_alt_train_ae.py \
            --data_folder /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/data/dataset_1 \
            --config /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
            --binning_file /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
            --load \
            --no_early_stop \
            --max_epochs $epoch \
            --layer_sizes $layer_size_unet \
            --learning_rate $learning_rate \
            --save_folder_append "grey_hyper/static_$str_lsu"e"$epoch"lr"$learning_rate"
        end=`date +%s`
        echo "training_time_seconds_e$epoch="$((end-start)) >> "/net/projects/fermi-stimulations/ae_models/dataset1_phot_AE/grey_hyper/static_$str_lsu"e"$epoch"lr"$learning_rate/training_time.txt"
    done
done

for lsu in "${layer_size_unet[@]}"
do
    str_lsu="${lsu//" "/"_"}"
    for epoch in $(seq $epoch_start_eval $epochs_per_eval $epoch_max)
    do
        python3 scripts/autoencoder/grey_alt_ae_eval.py \
            --data_folder /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/data/dataset_1 \
            --config /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
            --binning_file /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
            --model_loc "/net/projects/fermi-stimulations/ae_models/dataset1_phot_AE/grey_hyper/static_$str_lsu"e"$epoch"lr"$learning_rate/final.pth" \
            --model AE \
            --layer_sizes $layer_size_unet \
            --sample \
            --save_folder_append "grey_hyper/static_$str_lsu"e"$epoch"lr"$learning_rate"
        python3 scripts/autoencoder/grey_alt_ae_eval.py \
            --data_folder /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/data/dataset_1 \
            --config /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/configs/config_dataset1_photon.json \
            --binning_file /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/CaloChallenge/code/binning_dataset_1_photons.xml \
            --model_loc "/net/projects/fermi-stimulations/ae_models/dataset1_phot_AE/grey_hyper/static_$str_lsu"e"$epoch"lr"$learning_rate/final.pth" \
            --model AE \
            --layer_sizes $layer_size_unet \
            --save_folder_append "grey_hyper/static_$str_lsu"e"$epoch"lr"$learning_rate"
        python3 CaloChallenge/code/evaluate.py \
            --input_file "/net/projects/fermi-stimulations/ae_models/dataset1_phot_AE/grey_hyper/static_$str_lsu"e"$epoch"lr"$learning_rate/generated_dataset1_phot_AE.h5" \
            --reference_file /net/projects/fermi-1/doug/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/data/dataset_1/dataset_1_photons_2.hdf5 \
            --dataset "1-photons" \
            --output_dir "/net/projects/fermi-1/doug/sbatch_scripts/grey_hyper_plots/static_$str_lsu"e"$epoch"lr"$learning_rate"
    done
done
