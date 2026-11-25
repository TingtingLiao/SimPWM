#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -c 64
#SBATCH --mem=480G
#SBATCH -t 12:00:00
# GR1 IC-LoRA training with LTX-2's trainer, DDP over the nodes of the job (one accelerate launcher per node).
# usage (from the repo root): sbatch -N 4 scripts/gr1/train.sbatch CONFIG_YAML [ACCEL_YAML=configs/accelerate/ddp_4node_4gpu.yaml]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"  # sbatch runs a copy of this file
CFG=$(realpath "$1"); ACC=$(realpath "${2:-$REPO/configs/accelerate/ddp_4node_4gpu.yaml}")
LTX=$DEPS/LTX-2
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
PORT=$((20000 + SLURM_JOB_ID % 20000))
export SIMPWM_REF_POS_FIX=1
srun --kill-on-bad-exit=1 bash -c "cd '$LTX/packages/ltx-trainer' && unset PYTHONPATH && \
  '$LTX/.venv/bin/accelerate' launch --config_file '$ACC' --machine_rank \$SLURM_NODEID \
  --main_process_ip $MASTER --main_process_port $PORT scripts/train.py '$CFG'"
