#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH -c 64
#SBATCH --mem=800G
#SBATCH -o logs/%x-%j.log
# LTX IC-LoRA training on one 8-GPU node with LTX-2's trainer (config from simpwm/video/make_config.py).
#   ACCEL_YAML  configs/accelerate/ddp_8gpu.yaml (97 frames fit one 80 GB GPU per process) or fsdp_8gpu.yaml (241 frames)
# Submit from the repository root.
# usage: sbatch -t 2-00:00:00 scripts/droid/train.sbatch CONFIG_YAML ACCEL_YAML
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
CFG=$(realpath "$1") ACC=$(realpath "$2")
cd "$DEPS/LTX-2/packages/ltx-trainer"
unset PYTHONPATH
"$DEPS/LTX-2/.venv/bin/accelerate" launch --config_file "$ACC" scripts/train.py "$CFG" 2>&1 \
  | { grep --line-buffered -v 'leaked\|nanobind' || true; }
