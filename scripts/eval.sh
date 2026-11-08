#!/usr/bin/env bash
# Closed-loop evaluation in CARLA 0.9.15 (one CARLA server per GPU).
#
#   bash scripts/eval.sh <benchmark> <checkpoint> [options]
#
#   benchmark:    bench2drive | carla_f | b2d_c | b2d_adv
#   checkpoint:   a released model (<model>/<checkpoint>, config.yaml next to it) or a
#                 training checkpoint work_dirs/safedrive_vla/<experiment>/<timestamp>/checkpoints/epoch=013.ckpt
#   --gpu-ids 0,1,...           GPUs to use (default: all visible GPUs)
#   --nav-signal NAME           Bench2Drive navigation signal: target_point (default) | command
#   --routes-dir DIR            route files to evaluate (default: benchmark/data/<benchmark>)
#   --output-dir DIR            default: work_dirs/eval/<benchmark>/<experiment>
#   --save-viz                  save front-camera images with the predictions
#   --speed-track               CARLA-F: track the target speed parsed from the instruction
#                               while the model drives in strict mode (SAFEDRIVE_SPEED_TRACK)
#   --debug-log                 write agent_debug.json (timings, mode, speeds) per route
#   --route-walltime S          kill and retry a route after S wall-clock seconds (default 14400).
#                               B2D-C routes where the ego waits can need hours when several
#                               routes share a GPU; 3600 s killed them over and over.
#
# Requires CARLA_ROOT. Interrupted runs resume: completed routes are skipped.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BENCHMARK="${1:?usage: $0 <benchmark> <checkpoint> [options]}"
CHECKPOINT="${2:?usage: $0 <benchmark> <checkpoint> [options]}"
shift 2
GPU_IDS=""
OUTPUT_DIR=""
ROUTES_DIR="benchmark/data/${BENCHMARK}"
export SAFEDRIVE_NAV_SIGNAL="target_point"
export SAFEDRIVE_SAVE_VIZ="0"
export SAFEDRIVE_SPEED_TRACK="${SAFEDRIVE_SPEED_TRACK:-0}"
export SAFEDRIVE_DEBUG_LOG="${SAFEDRIVE_DEBUG_LOG:-0}"
ROUTE_WALLTIME=14400
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-ids)    GPU_IDS="$2"; shift 2 ;;
    --nav-signal) SAFEDRIVE_NAV_SIGNAL="$2"; shift 2 ;;
    --routes-dir) ROUTES_DIR="$(realpath "$2")"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --save-viz)   SAFEDRIVE_SAVE_VIZ="1"; shift ;;
    --speed-track) SAFEDRIVE_SPEED_TRACK="1"; shift ;;
    --debug-log)  SAFEDRIVE_DEBUG_LOG="1"; shift ;;
    --route-walltime) ROUTE_WALLTIME="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

case "${BENCHMARK}" in
  bench2drive) AGENT=team_code/agent_safedrive.py ;;
  carla_f)     AGENT=team_code/agent_safedrive_carla_f.py ;;
  b2d_c)       AGENT=team_code/agent_safedrive_b2d_c.py ;;
  b2d_adv)     AGENT=team_code/agent_safedrive_b2d_c.py ;;
  *) echo "Unknown benchmark: ${BENCHMARK}" >&2; exit 1 ;;
esac
: "${CARLA_ROOT:?set CARLA_ROOT to the CARLA 0.9.15 installation}"
[[ -e "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
if [[ -z "${GPU_IDS}" ]]; then
  GPU_IDS="$(seq -s, 0 $(( $(nvidia-smi -L | grep -c '^GPU ') - 1 )))"
fi
if [[ -z "${OUTPUT_DIR}" ]]; then
  CKPT_DIR="$(dirname "$(realpath "${CHECKPOINT}")")"
  if [[ -f "${CKPT_DIR}/config.yaml" ]]; then
    EXPERIMENT="$(basename "${CKPT_DIR}")"                            # released model
  else
    EXPERIMENT="$(basename "$(dirname "$(dirname "${CKPT_DIR}")")")"  # training run
  fi
  OUTPUT_DIR="work_dirs/eval/${BENCHMARK}/${EXPERIMENT}"
  [[ "${BENCHMARK}" == bench2drive && "${SAFEDRIVE_NAV_SIGNAL}" == command ]] && OUTPUT_DIR="${OUTPUT_DIR}_command"
fi
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
echo "speed_track=${SAFEDRIVE_SPEED_TRACK} debug_log=${SAFEDRIVE_DEBUG_LOG} save_viz=${SAFEDRIVE_SAVE_VIZ}"

python tools/run_distributed_eval.py \
  --routes-dir "${ROUTES_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --agent "${AGENT}" \
  --agent-config "$(realpath "${CHECKPOINT}")" \
  --gpu-ids "${GPU_IDS}" \
  --base-tm-port 8000 \
  --route-walltime "${ROUTE_WALLTIME}"

if [[ "${BENCHMARK}" == carla_f ]]; then
  python benchmark/metrics/carla_f_metrics.py "${OUTPUT_DIR}" --routes-dir "${ROUTES_DIR}"
else
  python benchmark/metrics/bench2drive_metrics.py "${OUTPUT_DIR}" --routes-dir "${ROUTES_DIR}"
fi
