#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -c 64
#SBATCH --mem=480G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.log
# Generate videos with MiniMax-H3 Ref2VA + a SimPWM-H3 LoRA from the encodings of scripts/h3/encode.sbatch: one video
# per GPU at a time, the model sharded over the GPUs of the node (4 x 80 GB: 42 GB peak, ~27 min per video on A100).
# Writes OUT/NAME/<idx>.mp4 (158 frames, 24 fps, 1280x704) and <idx>_15fps.mp4 (the 97-step clip grid); existing
# videos are skipped. Submit from the repository root.
# usage: sbatch scripts/h3/generate.sbatch CACHE OUT NAME LORA|none IDS_FILE
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
source "$REPO/scripts/h3/env.sh"
cd "$REPO"
torchrun --standalone --nproc-per-node="$(nvidia-smi -L | wc -l)" -m simpwm.h3.sample --cache "$1" --out "$2" \
  --variant "$3:$4:pin" --names "@$5"
