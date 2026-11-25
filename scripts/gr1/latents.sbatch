#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 08:00:00
# LTX latents (video + text + RGB proxy reference) of shard TASK of a GR1 dataset (LTX-2's process_dataset.py):
# 800x480x97 bucket (1280x768 videos stretched; 800x480 references at downscale factor 1), 10 fps.
# PRE/data/{latents,conditions,ref_rgb} is the trainer's preprocessed_data_root. Existing outputs are skipped.
# usage (from the repo root): sbatch -a 0-K-1 scripts/gr1/latents.sbatch DS K PRE [WxHxF=800x480x97]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"  # sbatch runs a copy of this file
DS=$(realpath "$1"); K=$2; PRE=$(realpath -m "$3"); B=${4:-800x480x97}; T=${SLURM_ARRAY_TASK_ID:-0}
LTX=$DEPS/LTX-2
read -r MODEL GEMMA < <("$PY" -c "from simpwm.video.make_config import GEMMA, MODEL; print(MODEL, GEMMA)")
mkdir -p "$PRE/rgb" "$PRE/data"
awk -v k="$K" -v t="$T" '(NR-1)%k==t' "$DS/dataset.jsonl" > "$DS/.part_$T.jsonl"
ln -sfn "$PRE/rgb/latents" "$PRE/data/latents"; ln -sfn "$PRE/rgb/conditions" "$PRE/data/conditions"
ln -sfn "$PRE/rgb/reference_latents" "$PRE/data/ref_rgb"
cd "$LTX/packages/ltx-trainer"; unset PYTHONPATH
"$LTX/.venv/bin/python" scripts/process_dataset.py "$DS/.part_$T.jsonl" --resolution-buckets "$B" \
  --model-path "$MODEL" --text-encoder-path "$GEMMA" --reference-downscale-factor 1 --reshape-mode stretch --fps 10 \
  --skip-audio --output-dir "$PRE/rgb"
