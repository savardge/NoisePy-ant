#!/bin/bash
#SBATCH --job-name=plotCR  # create a name for your job
#SBATCH --partition=geo #,glados12,glados16
#SBATCH --ntasks=1              # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --nodes=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output="outslurm/slurm-%j-%x.out"

python plot_substacks.py
