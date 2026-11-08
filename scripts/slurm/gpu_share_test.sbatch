#!/usr/bin/env bash
# How many evaluations can share one A100? Runs the same route as 1, 2 and 3
# simultaneous evaluations on one GPU (one after the other) and reports, per setting,
# the simulation speed (game seconds per wall second), wall time per route,
# peak GPU memory and the driving score of every copy (copies should agree).
#
#   sbatch scripts/slurm/gpu_share_test.sbatch
#   GROUPS_LIST="1 2" ROUTE=<route.xml> sbatch scripts/slurm/gpu_share_test.sbatch
#
# Output: logs/slurm/gpu_share-<jobid>.out (summary at the end; send it back),
# results in work_dirs/gpu_share/<jobid>/k<N>/. Submit from the repository root.
#SBATCH --job-name=gpu_share
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:30:00
#SBATCH --partition=dev-a100-40
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=logs/slurm/%x-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"

export CARLA_ROOT="${CARLA_ROOT:-/projects/EEHPC-DEV-2026D07-102/safedrive/carla0915}"
CONDA_ROOT="${CONDA_ROOT:-/projects/EEHPC-DEV-2026D07-102/dreamer/dependency/miniconda3}"
export PATH="${CONDA_ROOT}/envs/${ENV_NAME:-safedrive}/bin:${PATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/projects/EEHPC-DEV-2026D07-102/safedrive/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"

ROUTE="${ROUTE:-benchmark/dev_routes/first_route/bench2drive_50_language_rebuilt_turn_straight.xml}"
CKPT="${CKPT:-ckpts/safedrive-vla/safedrive_vla/model.ckpt}"
GROUPS_LIST="${GROUPS_LIST:-1 2 3}"
ROOT="work_dirs/gpu_share/${SLURM_JOB_ID:-manual}"
mkdir -p "$ROOT"

echo "node $(hostname)  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "route $ROUTE"; echo "groups $GROUPS_LIST"

for K in $GROUPS_LIST; do
  D="$ROOT/k$K"; mkdir -p "$D/routes"
  for i in $(seq 1 "$K"); do
    cp "$ROUTE" "$D/routes/$(basename "${ROUTE%.xml}")_copy$i.xml"
  done
  GPUS="$(printf '0,%.0s' $(seq 1 "$K"))"; GPUS="${GPUS%,}"
  echo; echo "========== k=$K  (gpu-ids $GPUS)  $(date +%T) =========="
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 > "$D/gpu.csv" &
  SMI=$!
  START=$(date +%s)
  bash scripts/eval.sh carla_f "$CKPT" --routes-dir "$D/routes" --output-dir "$D/eval" --gpu-ids "$GPUS"
  echo "eval.sh exit $?  wall $(( $(date +%s) - START )) s"
  kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
  echo $(( $(date +%s) - START )) > "$D/wall_s"
  sleep 10   # let the GPU memory drain before the next setting
done

echo; echo "========== SUMMARY =========="
python - "$ROOT" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
print(f"{'k':>2} {'wall_s':>7} {'peak_MiB':>9} {'mean_util':>9} | per copy: game_s/sys_s  sys_s  score  status")
for d in sorted(root.glob("k*"), key=lambda p: int(p.name[1:])):
    mem, util = [], []
    try:
        for row in csv.reader(open(d / "gpu.csv")):
            mem.append(float(row[1])); util.append(float(row[2]))
    except Exception:
        pass
    wall = (d / "wall_s").read_text().strip() if (d / "wall_s").exists() else "?"
    copies = []
    for res in sorted((d / "eval" / "res").glob("*_res.json")):
        try:
            rec = json.loads(res.read_text())["_checkpoint"]["records"][0]
            m = rec["meta"]; sc = rec["scores"]
            ratio = m["duration_game"] / max(m["duration_system"], 1e-6)
            copies.append(f"{ratio:.3f} {m['duration_system']:.0f} {sc.get('score_composed', 0):.1f} {rec.get('status','')}")
        except Exception as e:
            copies.append(f"unreadable ({e})")
    peak = max(mem) if mem else float("nan")
    mu = sum(util) / len(util) if util else float("nan")
    print(f"{d.name[1:]:>2} {wall:>7} {peak:>9.0f} {mu:>9.1f} | " + " ; ".join(copies))
print("\nThroughput gain of k groups = k * (game_s/sys_s at k) / (game_s/sys_s at 1).")
PY
