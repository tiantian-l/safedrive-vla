#!/usr/bin/env bash
# E17 ([26]) paired junction data: PDM-Lite (carla_garage fork, branch leaderboard_2) on the routes of
# work_dirs/pair_junctions/check/collect.json, see scripts/data/collect_pairs.py. CARLA is started per slot
# like the Bench2Drive evaluator in scripts/eval.sh does (CarlaUE4.sh -RenderOffScreen -graphicsadapter).
# Arguments go to `collect_pairs.py run`. Submit from the repository root; resubmitting resumes.
#
#   sbatch -p dev-a100-40 --gpus=1 --time=01:00:00 -J E17-collect-try -D $PWD scripts/slurm/collect_pairs.sbatch \
#       --groups <group> --gpu-ids 0,0,0
#
#SBATCH --job-name=E17-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export CARLA_ROOT="${CARLA_ROOT:-/projects/EEHPC-DEV-2026D07-102/safedrive/carla0915}"
CONDA_ROOT="${CONDA_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3}"
export PATH="${CONDA_ROOT}/envs/${ENV_NAME:-safedrive}/bin:${PATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
# Only the NVIDIA Vulkan driver, so that CARLA never renders on lavapipe (CPU).
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
unset PYTHONPATH
nvidia-smi --query-gpu=index,name,memory.total --format=csv
git -C "${GARAGE:-../carla_garage}" log --oneline -1 || true
python -u scripts/data/collect_pairs.py run --garage "${GARAGE:-../carla_garage}" "$@"
python scripts/data/collect_pairs.py report "$@" 2>/dev/null | tail -n 40 || true
