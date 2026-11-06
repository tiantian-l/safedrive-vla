#!/bin/bash
# CARLA 0.9.15 smoke test on one A100: can the cluster run CARLA headless?
#   sbatch scripts/slurm/carla_smoke_test.sbatch
#   LOAD_TOWN=none sbatch scripts/slurm/carla_smoke_test.sbatch   # skip the large-map test
# Checks, in order: GPU, Vulkan, internet, CARLA start, Python client connection,
# loading a large town (Town12, needs AdditionalMaps), GPU memory of CARLA.
# Send the whole slurm-carla-smoke-<jobid>.out back for review.
#SBATCH --job-name=carla_smoke
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=dev-a100-80
#SBATCH --account=eehpc-dev-2026d07-102g
#SBATCH --output=slurm-carla-smoke-%j.out
set -uo pipefail

PROJECT=/projects/EEHPC-DEV-2026D07-102
export CARLA_ROOT="${CARLA_ROOT:-$PROJECT/safedrive/carla0915}"
export CONDA_ROOT="${CONDA_ROOT:-$PROJECT/dreamer/dependency/miniconda3}"
export ENV_NAME="${ENV_NAME:-safedrive}"
PY="${CONDA_ROOT}/envs/${ENV_NAME}/bin/python"
PORT="${PORT:-2000}"
LOAD_TOWN="${LOAD_TOWN:-Town12}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# The node registers four Vulkan drivers (NVIDIA, Intel, Radeon and lavapipe, a CPU
# software renderer). Expose only NVIDIA so that CARLA cannot pick a CPU renderer.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
export VK_DRIVER_FILES="$VK_ICD_FILENAMES"
UE_LOG="$CARLA_ROOT/CarlaUE4/Saved/Logs/CarlaUE4.log"

step() { echo; echo "========== $* =========="; }

step "1. Node and GPU"
hostname; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

step "2. Vulkan (CARLA renders with Vulkan)"
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1
echo "VK_ICD_FILENAMES=$VK_ICD_FILENAMES"; cat "$VK_ICD_FILENAMES"
command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>&1 | head -30 || echo "vulkaninfo not installed (not fatal)"
ldconfig -p | grep -E "libvulkan|libGLX_nvidia|libEGL_nvidia" || echo "WARNING: no Vulkan/NVIDIA GL libraries found by ldconfig"

step "3. Internet from the compute node"
curl -sS -m 10 -o /dev/null -w "huggingface.co HTTP %{http_code}\n" https://huggingface.co || echo "NO internet on compute nodes -> download everything on the login node, set HF_HUB_OFFLINE=1"

step "4. Python environment"
"$PY" -c "import carla, torch; print('carla client OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || echo "WARNING: python env $PY not ready"

step "5. Start CARLA headless on port $PORT"
test -x "$CARLA_ROOT/CarlaUE4.sh" || { echo "FATAL: $CARLA_ROOT/CarlaUE4.sh not found"; exit 1; }
JOB="${SLURM_JOB_ID:-local}"
SERVER_LOG="carla_smoke_server_${JOB}.log"
GPU_LOG="carla_smoke_gpu_${JOB}.csv"
# GPU utilisation / memory every 15 s: shows whether CARLA is rendering at all.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 15 > "$GPU_LOG" 2>&1 &
MON_PID=$!
"$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -prefernvidia -carla-rpc-port="$PORT" -graphicsadapter=0 > "$SERVER_LOG" 2>&1 &
CARLA_PID=$!
cleanup() {
  step "Server log (last 40 lines)"; tail -40 "$SERVER_LOG"
  step "Unreal Engine log: Vulkan / GPU / errors"
  grep -iE "vulkan|rhi|adapter|gpu|error|fatal|crash|exit" "$UE_LOG" 2>/dev/null | head -60 || echo "no UE log at $UE_LOG"
  cp "$UE_LOG" "carla_smoke_ue_${JOB}.log" 2>/dev/null
  step "GPU monitor"; cat "$GPU_LOG"
  kill "$MON_PID" 2>/dev/null
  kill -- -"$CARLA_PID" 2>/dev/null; pkill -f "carla-rpc-port=$PORT" 2>/dev/null
}
trap cleanup EXIT
for i in $(seq 1 24); do
  if ! kill -0 "$CARLA_PID" 2>/dev/null; then echo "FATAL: CARLA exited during start-up"; exit 1; fi
  if (echo > /dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then echo "[$(date +%T)] CARLA is listening on port $PORT after $((i * 5)) s"; break; fi
  sleep 5
done
(echo > /dev/tcp/127.0.0.1/"$PORT") 2>/dev/null || { echo "FATAL: CARLA not listening on port $PORT after 120 s"; exit 1; }

step "6. Python client: default map + camera first, then $LOAD_TOWN"
timeout 2700 "$PY" -u - "$PORT" "$LOAD_TOWN" <<'PYEOF'
import sys, time
import carla
T0 = time.time()
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:5.0f}s] {msg}", flush=True)

def drive(world, ticks, label):
    """Ego vehicle + front camera in synchronous mode; returns camera frames received."""
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
    world.apply_settings(settings)
    lib = world.get_blueprint_library()
    ego = world.spawn_actor(lib.find("vehicle.lincoln.mkz_2020"), world.get_map().get_spawn_points()[0])
    cam_bp = lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1024"); cam_bp.set_attribute("image_size_y", "512")
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-1.5, z=2.0)), attach_to=ego)
    frames = []
    cam.listen(lambda img: frames.append(img.frame))
    log(f"{label}: ego + camera spawned, ticking")
    t0 = time.time()
    for k in range(ticks):
        world.tick()
        if k in (0, 9) or (k + 1) % 50 == 0:
            log(f"{label}: tick {k + 1}/{ticks}, camera frames {len(frames)}")
    dt = time.time() - t0
    log(f"{label}: {ticks} ticks in {dt:.1f}s ({ticks / dt:.1f} ticks/s), camera frames {len(frames)}")
    cam.stop(); cam.destroy(); ego.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    return len(frames)

port, town = int(sys.argv[1]), sys.argv[2]
client = carla.Client("127.0.0.1", port)
client.set_timeout(20.0)
for attempt in range(1, 16):
    try:
        world = client.get_world()
        break
    except RuntimeError as e:
        log(f"connect attempt {attempt} failed: {e}")
        time.sleep(5)
else:
    log("FATAL: could not connect to CARLA"); sys.exit(1)
log(f"connected: server {client.get_server_version()}, map {world.get_map().name}")

client.set_timeout(120.0)
ok_small = drive(world, 100, "A default map") > 0
log("A default map: " + ("rendering OK" if ok_small else "FATAL: no camera frames, rendering is broken"))
if not ok_small:
    sys.exit(1)

if town.lower() != "none":
    client.set_timeout(1800.0)
    log(f"B loading {town} (large map, can take many minutes) ...")
    t0 = time.time()
    world = client.load_world(town)
    log(f"B loaded {world.get_map().name} in {time.time() - t0:.0f}s")
    client.set_timeout(120.0)
    if drive(world, 50, f"B {town}") == 0:
        log(f"FATAL: no camera frames on {town}"); sys.exit(1)
log("SMOKE TEST PASSED")
PYEOF
echo "python exit code: $?"

step "7. GPU memory now"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=memory.used --format=csv
