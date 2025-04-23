#!/bin/bash

#SBATCH --job-name=vgmap # create a name for your job
#SBATCH --partition=public-bigmem,shared-bigmem,public-cpu,shared-cpu
#SBATCH --output="outslurm/slurm-%A_%a-%x.out"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-64 #2-64 

#module load MATLAB

unset DISPLAY

TC=`sed -n "${SLURM_ARRAY_TASK_ID}p" Tc.list`
echo "Period: $TC"

matlab -batch "main_riehen($TC)"
matlab -batch "main_aargau($TC)"

#matlab -batch "main_hautesorne_rayleigh_1($TC)"

#matlab -batch "main_hautesorne_love_1($TC)"

