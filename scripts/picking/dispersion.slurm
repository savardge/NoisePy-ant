#!/bin/bash
#SBATCH --job-name=dc-aargau  # create a name for your job
#SBATCH --partition=shared-cpu,public-cpu,public-bigmem,shared-bigmem
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=11:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/dev/null 
##SBATCH --output="outslurm/slurm-%A_%a_%x.out"
#SBATCH --array=0-221%60 #max 22146

fname="text_files/stack_filelist_CH-AA_norm.txt"
for k in `seq 0 99`; do
    ROW=$(( ${SLURM_ARRAY_TASK_ID} * 100 + $k ))
    if [ "$ROW" -eq "0" ]; then
        continue
    fi
    sfile=`sed -n "${ROW}p" $fname`
    echo $k $ROW $sfile

    python dispersion_curves_V2.py $sfile
    
done
