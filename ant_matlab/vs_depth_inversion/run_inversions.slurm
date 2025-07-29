#!/bin/bash

#SBATCH --job-name=swantG_riehen_run1lvz
#SBATCH --partition=public-cpu,shared-cpu
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=8:00:00
#SBATCH --output="outslurm/slurm-%A_%a_stepG_aargau.out"
#SBATCH --array=1-9000:500

#module load MATLAB/2021a
#export LD_LIBRARY_PATH=/home/share/cdff/Geopsy/lib:/home/users/s/savardg/anaconda3/envs/geopsy/lib:${LD_LIBRARY_PATH}


# *** STEP G: 3D DEPTH INVERSION
echo "Launching Depth inversion"

IND=${SLURM_ARRAY_TASK_ID}
NUM=499
echo "Starting index: ${IND}, number of indices to process: ${NUM}"
#matlab -batch "C_launch_inversion_riehen($IND,$NUM)"
matlab -batch "C_launch_inversion_riehen_run1($IND,$NUM)"
