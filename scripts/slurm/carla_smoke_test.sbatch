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
#SBATCH --time=00:30:00
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
export PYTHONNOUSERSITE=1

step() { echo; echo "========== $* =========="; }

step "1. Node and GPU"
hostname; date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

step "2. Vulkan (CARLA renders with Vulkan)"
ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1
command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>&1 | head -30 || echo "vulkaninfo not installed (not fatal)"
ldconfig -p | grep -E "libvulkan|libGLX_nvidia|libEGL_nvidia" || echo "WARNING: no Vulkan/NVIDIA GL libraries found by ldconfig"

step "3. Internet from the compute node"
curl -sS -m 10 -o /dev/null -w "huggingface.co HTTP %{http_code}\n" https://huggingface.co || echo "NO internet on compute nodes -> download everything on the login node, set HF_HUB_OFFLINE=1"

step "4. Python environment"
"$PY" -c "import carla, torch; print('carla client OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())" || echo "WARNING: python env $PY not ready"

step "5. Start CARLA headless on port $PORT"
test -x "$CARLA_ROOT/CarlaUE4.sh" || { echo "FATAL: $CARLA_ROOT/CarlaUE4.sh not found"; exit 1; }
"$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -carla-rpc-port="$PORT" -graphicsadapter=0 > carla_smoke_server_${SLURM_JOB_ID}.log 2>&1 &
CARLA_PID=$!
trap 'kill -- -$CARLA_PID 2>/dev/null; pkill -f "carla-rpc-port=$PORT" 2>/dev/null' EXIT
sleep 20
if ! kill -0 "$CARLA_PID" 2>/dev/null; then
  echo "FATAL: CARLA exited during start-up. Last lines of its log:"; tail -50 carla_smoke_server_${SLURM_JOB_ID}.log; exit 1
fi

step "6. Python client: connect, load $LOAD_TOWN, run 100 synchronous ticks"
"$PY" - "$PORT" "$LOAD_TOWN" <<'PYEOF'
import sys, time
import carla
port, town = int(sys.argv[1]), sys.argv[2]
client = carla.Client("localhost", port)
client.set_timeout(300.0)
t0 = time.time()
for attempt in range(12):
    try:
        world = client.get_world(); break
    except RuntimeError as e:
        print(f"waiting for server ({e})"); time.sleep(10)
else:
    sys.exit("FATAL: could not connect to CARLA")
print(f"connected after {time.time()-t0:.0f}s, server {client.get_server_version()}, map {world.get_map().name}")
if town.lower() != "none":
    t0 = time.time()
    world = client.load_world(town)
    print(f"loaded {world.get_map().name} in {time.time()-t0:.0f}s")
settings = world.get_settings()
settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
world.apply_settings(settings)
bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
spawn = world.get_map().get_spawn_points()[0]
ego = world.spawn_actor(bp, spawn)
cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1024"); cam_bp.set_attribute("image_size_y", "512")
frames = []
cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-1.5, z=2.0)), attach_to=ego)
cam.listen(lambda img: frames.append(img.frame))
t0 = time.time()
for _ in range(100):
    world.tick()
dt = time.time() - t0
print(f"100 ticks in {dt:.1f}s ({100/dt:.1f} ticks/s), camera frames received: {len(frames)}")
cam.destroy(); ego.destroy()
settings.synchronous_mode = False
world.apply_settings(settings)
print("SMOKE TEST PASSED" if frames else "FATAL: no camera frames (rendering is broken)")
PYEOF

step "7. GPU memory used by CARLA"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-gpu=memory.used --format=csv
