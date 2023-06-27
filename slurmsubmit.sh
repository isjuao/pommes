#!/bin/bash --login
#SBATCH --job-name=MRP
#SBATCH --partition=uoa-gpu
#SBATCH --time=500:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=t08io22@abdn.ac.uk

module load python
current_date=$(date +"%Y-%m-%d")
infix="_job-"
python mrp_tl.py -n > "log/log_$current_date$infix$SLURM_JOB_ID.out"