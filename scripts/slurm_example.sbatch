#!/bin/bash
# Example SLURM job. Fill in the partition / GPU type / environment of your cluster.
# One benchmark per job; the whole job must stay on one GPU (the cost table belongs to that GPU).
# Rough time with REPEATS=6 (timing + FLOP records): A40 GSM8K ~11 h, MATH-500 ~19 h; H100 GSM8K ~5.5 h, MATH-500 ~9.5 h.
#SBATCH --job-name=nld-fused
#SBATCH --partition=<PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/%x_%j.out
#SBATCH --error=runs/%x_%j.err
# On nodes shared with other jobs we saw the baseline slow down more than the fused method (it runs more forwards),
# so an exclusive node gives cleaner timings:
##SBATCH --exclusive

source <PATH_TO_VENV>/bin/activate
# export HF_HOME=<HF_CACHE_DIR>
# export NLD_MODEL=<LOCAL_CHECKPOINT_DIR>   # if the checkpoint is already downloaded

DATASET=${DATASET:-gsm8k}
bash scripts/run.sh "$DATASET" "runs/${DATASET}_${SLURM_JOB_ID}"
