#!/bin/bash
#SBATCH --job-name=get-cwt  # create a name for your job
#SBATCH --partition=shared-cpu
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem-per-cpu=4G         # memory per cpu-core
#SBATCH --time=11:00:00          # total run time limit (HH:MM:SS)
###SBATCH --output=/dev/null   # this option does not generate an output log file from slurm. Do this when you're sure the code won't need debugging to minimize creation of useless log files
#SBATCH --output="outslurm/slurm-%A_%a_%x.out" # this specifies the path of the output log file from slurm. Make sure the directory "outslurm" exists!
#SBATCH --array=0-221%80    # The number after the % specifies the max number of simultaneous jobs. This is too prevent overwhelming the cluster with a lot of I/O operations. Around 100 is a good number

# The array job numbers in "SBATCH --array" start at 0, and goes up to NLINES/100 if processing chunks of 100 files, where NLINES is the number of lines in fname
# You can get the number of lines with the unix command ` wc -l filename.txt` (replace "filename.txt" with your actual file name)
# For example, if your file fname has 1000 lines, then --array=0-10

# Specify the text file with the the full path to the H5 files. One file name per line. Must be the complete path
fname="stack_filelist_CH-AA_norm.txt"

# In this example, this array job element will run the codes for 100 lines of the text file, starting at line SLURM_ARRAY_TASK_ID*100
for k in `seq 0 99`; do  # seq 0 99 generates a list of integers from 0 to 99
    ROW=$(( ${SLURM_ARRAY_TASK_ID} * 100 + $k )) # Get the line number.
    if [ "$ROW" -eq "0" ]; then  # Skip when it equals 0 because line number starts at 1
        continue
    fi
    sfile=`sed -n "${ROW}p" $fname`  # Get the file for the line number ROW
    echo "Line number $ROW (iteration $k / 100 for this array job element: ${SLURM_ARRAY_TASK_ID}). File name is: $sfile "

    # Give the filepath as input argument to the scripts
    python CWT_h5_to_dat_1file.py $sfile
    python TimeShift_PhaseSpectrograms_h5_to_dat_1file.py $sfile

done
