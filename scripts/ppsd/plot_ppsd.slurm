#!/bin/bash
#SBATCH --job-name=ppsd_riehen # create a name for your job
#SBATCH --partition=shared-cpu,public-cpu,public-bigmem,shared-bigmem
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=16G         # memory per cpu-core
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output="outslurm/slurm-%A_%a-%x.out"
#SBATCH --array=2-198
#SBATCH --mail-type=START,END
#SBATCH --mail-user=genevieve.savard@unige.ch

#unset DISPLAY  # uncomment if doing plotting

fname="station_locations_riehen.csv"
station=`sed -n "${SLURM_ARRAY_TASK_ID}p" $fname | awk -F, '{print $2}'` # if station name in 2nd column
echo $station 
configfile="ppsd_params.yaml"

python ppsd_1sta.py $station DPZ $configfile
#python ppsd_1sta.py $station DPN $configfile
#python ppsd_1sta.py $station DPE $configfile
