#!/bin/bash
#SBATCH --job-name=UA_MTQ
#SBATCH --time=00:30:00
# #SBATCH --time=04:00:00
#SBATCH --account=def-lbriand
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8g
# #SBATCH --array=1-270
#SBATCH --array=1-3
#SBATCH --output=/home/sepshr/scratch/sepshr/MLCSHE_UpdateArchive_results/slurm_out/%A_%a.out

echo "Starting task $SLURM_ARRAY_TASK_ID"

$config_id=$(expr $SLURM_ARRAY_TASK_ID % 27 + 1)

# to read from the list of configs
config_ver=$(sed -n "${config_id}p" configs.list)

echo "Running: $config_ver"


# bash min_alg_nsga2_one_version.sh $config_ver
bash run_iccea_mtq.sh $config_ver
