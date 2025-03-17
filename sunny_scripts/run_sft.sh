#!/bin/sh

#SBATCH -c 64 # Number of cores requested
#SBATCH -t 0-06:00 # Runtime in minutes
#SBATCH -p kempner_h100 # Partition to submit to
#SBATCH --mem=250G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:2
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this filehostname hostname
#SBATCH --account=kempner_dam_lab
#SBATCH --job-name=sft-MATH

module purge
module load Mambaforge
module load cuda cudnn
mamba activate verl

set -x 

cd /n/home05/sqin/wall/verl/verl/

# accelerate launch --config_file ../configs/accelerate.yaml train.py --config ../configs/oft-mix-4-cd.conf 

accelerate launch --config_file /n/home05/sqin/wall/verl/sunny_scripts/accelerate.yaml \
  trainer/sft_trainer.py

    # --ckpt /n/netscratch/dam_lab/Lab/sqin/outputs/oft/checkpoint-9500 \
    # --resume 
