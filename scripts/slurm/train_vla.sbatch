#!/usr/bin/env bash
# Slurm wrapper of scripts/train_vla.sh (one node). Activate the conda
# environment before submitting; add -p/-A/-t for your cluster, e.g.
#
#   sbatch -p <partition> scripts/slurm/train_vla.sbatch safedrive_vla
#
# Resubmit with RESUME=1 to continue from the last complete epoch.
#
#SBATCH --job-name=safedrive-vla
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# Same environment as eval.sbatch (11/20): compute nodes are offline.
CONDA_ROOT="${CONDA_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3}"
export PATH="${CONDA_ROOT}/envs/${ENV_NAME:-safedrive}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/projects/EEHPC-DEV-2026D07-102/safedrive/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
# The launcher starts one process per GPU itself (Lightning / torchrun);
# hide the Slurm task layout so it is not mistaken for an srun step.
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE SLURM_PROCID SLURM_LOCALID
NUM_GPUS="$(nvidia-smi -L | grep -c '^GPU ')" bash scripts/train_vla.sh "$@"
