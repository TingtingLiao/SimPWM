# Common environment of the launch scripts (sourced by scripts/*/*.sh and *.sbatch).
#   PY         python of the main environment (default: python on PATH)
#   HF_HOME    Hugging Face cache holding the perception / VLM weights (jobs run offline)
#   SAM3_PATH  local snapshot of facebook/sam3
#   SIMPWM_DEPS, SIMPWM_RUNS  external code and outputs (default: deps/, runs/), as DEPS and RUNS
#   CUDA_HOME  CUDA toolkit for nvdiffrast's JIT build (optional: without it the robot is ray-cast on the CPU)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PY=${PY:-python}
DEPS=${SIMPWM_DEPS:-$REPO/deps} RUNS=${SIMPWM_RUNS:-$REPO/runs}
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$(dirname "$(command -v "$PY")")${CUDA_HOME:+:$CUDA_HOME/bin}:$PATH  # ninja (+ nvcc) for nvdiffrast
mkdir -p "$REPO/logs"

# wait_all: wait for the background jobs and fail if one of them failed (a bare `wait` returns 0)
wait_all() { local rc=0 p; for p in $(jobs -p); do wait "$p" || rc=1; done; return $rc; }

# shard_ids FILE TASK N: the TASK-th of N interleaved shards of a whitespace-separated id list
shard_ids() { "$PY" -c "import sys; ids = open(sys.argv[1]).read().split(); print(' '.join(ids[int(sys.argv[2])::int(sys.argv[3])]))" "$@"; }
