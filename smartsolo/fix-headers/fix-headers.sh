#!/bin/bash
#SBATCH --array=1 #-189  # 1 to number of stations
#SBATCH --partition=shared-cpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output="outslurm/slurm-%A_%a.out"

# Data directory root
DATAROOT="/srv/beegfs/scratch/shares/cdff/MIGRATE/Tuscany_July2023/miniseed" 

# Path to program executable file
msmod=/home/share/cdff/msmod-1.2/msmod

# Get directory to process
datadir=`ls -d $DATAROOT/* | sort | sed -n "${SLURM_ARRAY_TASK_ID}p"`

# Create a log file where we keep track of directories processed so far.
logfile="msmod-output/fix-msmod_${SLURM_ARRAY_TASK_ID}.log"
rm -f $logfile # remove if it already exists
date >> $logfile # Add date and time at beginning of file for when script starts running.

echo "*** PROCESSING DIRECTORY *** = $datadir"

# Get net, sta and chan from an example filename in the directory
firstfile=`ls $datadir/*.miniseed | head -1` # extract the first file in the directory.
NET=`echo $firstfile | xargs -n1 basename | awk -F. '{print $1}'`
STA=`echo $firstfile | xargs -n1 basename | awk -F. '{print $2}'`
#LOC=`echo $firstfile | awk -F. '{print $3}'`
#CHAN=`echo $firstfile | awk -F. '{print $4}'`
#echo "NET=$NET, STA=$STA, CHAN=$CHAN"
echo "NET=$NET, STA=$STA"

# Now run msmod to change header to corresponding values in file name
#$msmod $datadir/*.miniseed -i --net $NET --sta $STA --chan $CHAN
$msmod $datadir/*.miniseed -i --net $NET --sta $STA
 
# Update log file
echo "all done" >> $logfile
date >> $logfile # Add date and time at end of file for when script finishes.
