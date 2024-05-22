#!/bin/bash
#
#SBATCH --job-name=fermi
#SBATCH --output=/net/projects/fermi-1/logs/victor/test_sbatch_eval.stdout
#SBATCH --error=/net/projects/fermi-1/logs/victor/test_sbatch_eval.stderr
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

cd /home/${USER}/2023-Autumn-Clinic-Fermi-CaloDiffusionPaper/
source /home/${USER}/.bashrc
source activate calo_diffusion
python3 scripts/evaluate.py \
    --input_file 'ND'\
    --reference_file 'ND'\
    --mode \
    --dataset '1-photons' \
    --output_dir \
    --ratio \
    --cls_n_layer \
    --cls_n_iters \
    --cls_n_hidden \
    --cls_dropout_probability \
    --cls_batch_size \
    --cls_n_epochs \
    --cls_lr \
    --save_mem