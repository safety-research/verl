#!/bin/sh

#SBATCH -c 16 # Number of cores requested
#SBATCH -t 0-07:00 # Runtime in minutes
#SBATCH -p kempner # Partition to submit to
#SBATCH --mem=64G # Memory per node in MB (see also --mem-per-cpu)
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o slurm_out/slurm-%j.out # Standard out goes to this file
#SBATCH -e slurm_out/slurm-%j.out # Standard err goes to this file
#SBATCH --account=kempner_barak_lab
#SBATCH --exclude=holygpu8a19604,holygpu8a19303,holygpu8a17402,holygpu8a17402
#SBATCH --job-name=annot_base


module purge
module load Mambaforge
module load cuda cudnn
module load gcc/12.2.0-fasrc01 
mamba activate verl

set -x 
cd /n/home05/sqin/wall/verl/analysis
pwd


python analyze_model_outputs.py