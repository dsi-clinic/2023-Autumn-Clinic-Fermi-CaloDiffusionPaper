#!/bin/bash
#
#SBATCH --mail-user=CNetID@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=fermi-sim
#SBATCH --output=./logs/%j.%N.stdout
#SBATCH --error=./logs/%j.%N.stderr
#SBATCH --chdir=/home/CNetID/PATH_TO_SCRIPTS_DIR
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
​
source /home/${USER}/.bashrc
source activate env_name 
python3 run.py 
