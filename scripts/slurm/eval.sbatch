#!/usr/bin/env bash
# Slurm wrapper of scripts/eval.sh (one node = 4 A100; use --gpu-ids 0,0,1,1,2,2,3,3 for two routes per GPU).
# CARLA_ROOT and the conda environment default to the project paths below; override
# them by exporting CARLA_ROOT / CONDA_ROOT / ENV_NAME before submitting, e.g.
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
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export CARLA_ROOT="${CARLA_ROOT:-/projects/EEHPC-DEV-2026D07-102/safedrive/carla0915}"
CONDA_ROOT="${CONDA_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3}"
export PATH="${CONDA_ROOT}/envs/${ENV_NAME:-safedrive}/bin:${PATH}"
export PYTHONNOUSERSITE=1
# Compute nodes have no internet: read models from the cache filled on the login node.
export HF_HOME="${HF_HOME:-/projects/EEHPC-DEV-2026D07-102/safedrive/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
# Only the NVIDIA Vulkan driver, so that CARLA never renders on lavapipe (CPU).
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
bash scripts/eval.sh "$@"
