#!/bin/bash
source /home/${USER}/.bashrc

#Note that this is not itself an sbatch script, but instead submits multiple jobs.

#Change this accordingly so we source activate the correct env
env="calo_diffusion"

dataset="2train" #Number of dataset. Current options: "1", "2train", "2eval"
layer_size_unet=( "64 64 64" "128 128 128 128" "128 128 128 128 128" )
learning_rate=( 0.0004 0.0001 0.00004 )

sed -i '12s/.*/source activate '$env'/' /net/projects/fermi-1/doug/sbatch_scripts/grey/autoscript_ae_ds$dataset.sh

for lsu in "${layer_size_unet[@]}"
do
    command="sed -i '14s/.*/layer_size_unet=( \"$lsu\" )/' /net/projects/fermi-1/doug/sbatch_scripts/grey/autoscript_ae_ds$dataset.sh"
    eval $command
    for lr in "${learning_rate[@]}"
    do
        sed -i '28s/.*/learning_rate='$lr'/' autoscript_ae_ds$dataset.sh
        sbatch_output=$(sbatch /net/projects/fermi-1/doug/sbatch_scripts/grey/autoscript_ae_ds$dataset.sh)
        echo $sbatch_output
    done
done
