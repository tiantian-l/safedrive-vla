#!/usr/bin/env bash
# Slurm wrapper of scripts/eval.sh (one node = 4 A100; use --gpu-ids 0,0,1,1,2,2,3,3 for two routes per GPU).
# Activate the conda environment and export CARLA_ROOT before submitting, e.g.
#
#   sbatch scripts/slurm/eval.sbatch carla_f <checkpoint> --gpu-ids 0,0,1,1,2,2,3,3
#
# Resubmitting the same command resumes the evaluation.
#
#SBATCH --job-name=safedrive-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=120
#SBATCH --mem=460G
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal-a100-80
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# Compute nodes have no internet: read models from the cache filled on the login node.
export HF_HOME="${HF_HOME:-/projects/EEHPC-DEV-2026D07-102/safedrive/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
bash scripts/eval.sh "$@"
