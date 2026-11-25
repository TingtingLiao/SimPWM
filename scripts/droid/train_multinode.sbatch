#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH -o logs/%x-%j.log
# LTX IC-LoRA DDP training over several nodes, one accelerate launcher per node (config from simpwm/video/make_config.py).
#   ACCEL_YAML  e.g. configs/accelerate/ddp_4node_4gpu.yaml: num_machines = nodes, num_processes = nodes x GPUs per node
# Submit from the repository root.
# usage: sbatch -N 4 --gres=gpu:4 --mem=480G -t 1-00:00:00 scripts/droid/train_multinode.sbatch CONFIG_YAML ACCEL_YAML
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
CFG=$(realpath "$1") ACC=$(realpath "$2")
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT=$((20000 + SLURM_JOB_ID % 20000))
srun --kill-on-bad-exit=1 bash -c "cd $DEPS/LTX-2/packages/ltx-trainer; unset PYTHONPATH; \
  $DEPS/LTX-2/.venv/bin/accelerate launch --config_file $ACC --machine_rank \$SLURM_NODEID \
  --main_process_ip $MASTER --main_process_port $PORT scripts/train.py $CFG" 2>&1 \
  | { grep --line-buffered -v 'leaked\|nanobind' || true; }
