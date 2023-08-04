#!/bin/bash

#SBATCH --job-name=F-run2  # create a name for your job
#SBATCH --partition=shared-cpu,public-cpu,public-bigmem,shared-bigmem
#SBATCH --output="outslurm/slurm-%A_%a-%x.out"
#SBATCH --partition=shared-cpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --nodes=1
###SBATCH --mem=18G
#SBATCH --mem-per-cpu=24G
#SBATCH --time=8:00:00
#SBATCH --array=1-54 #54:5

#module load MATLAB

# *** Run range of params on one period
#TC=`sed -n "${SLURM_ARRAY_TASK_ID}p" Tc.list`
#echo "Period: $TC"
#matlab -batch "B_inversion_TV_param_set_sigma_Lc($TC)"
#echo "Period: $TC"
#echo "*** Aargau ***"
#matlab -batch "B_inversion_TV_param_set_sigma_Lc_aargau($TC)"
#echo "*** Riehen ***"
#matlab -batch "B_inversion_TV_param_set_sigma_Lc_riehen($TC)"

# Run for one period and one set of params
TC=`sed -n "${SLURM_ARRAY_TASK_ID}p" Tc.list`
#echo "Period: $TC"
#echo "*** Aargau ***"
#echo "boostrap inversion"
#matlab -batch "D_compute_inversion_TV_1period_aargau_bootstrap($TC)"
echo "ZZ"
matlab -batch "D_compute_inversion_TV_1period_aargau_ZZ($TC)"
#echo "ZR"
#matlab -batch "D_compute_inversion_TV_1period_aargau_ZR($TC)"
#echo  "ZZ-ZR"
#matlab -batch "D_compute_inversion_TV_1period_aargau_ZZZR($TC)"
#echo "all4"
#matlab -batch "D_compute_inversion_TV_1period_aargau_all4($TC)"
echo "TT"
matlab -batch "D_compute_inversion_TV_1period_aargau_TT($TC)"
#echo "RZ"
#matlab -batch "D_compute_inversion_TV_1period_aargau_RZ($TC)"
echo "RR"
matlab -batch "D_compute_inversion_TV_1period_aargau_RR($TC)"

echo "*** Riehen ***"
echo "boostrap inversion"
#matlab -batch "D_compute_inversion_TV_1period_riehen_bootstrap($TC)"

#echo "*** Riehen ***"
#matlab -batch "D_compute_inversion_TV_1period_riehen($TC)"

# *** Run all periods with same given params
#matlab -batch "D_compute_inversion_TV_paramfix_allperiods_riehen()"
#matlab -batch "D_compute_inversion_TV_paramfix_allperiods_aargau()"
