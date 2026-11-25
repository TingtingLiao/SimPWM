#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.log
# LTX latents of shard TASK of a link-mode dataset (simpwm/video/dataset.py), with LTX-2's scripts/process_dataset.py:
#   rgb pass    dataset.jsonl       -> PRE/rgb/{latents,conditions,reference_latents}  (RGB proxy at 1/2 resolution)
#   depth pass  dataset_depth.jsonl -> PRE/depth/reference_latents (latents/, conditions/ link the rgb pass)
# PRE/data/{latents,conditions,ref_rgb,ref_depth} is then the trainer's preprocessed_data_root. Existing outputs are
# skipped, so shards share PRE and can be resubmitted. Submit from the repository root.
# usage: sbatch -a 0-31 scripts/droid/latents.sbatch DS_DIR 32 PRE WxHxF   (e.g. 1280x704x241)
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
DS=$(realpath "$1") K=$2 PRE=$(realpath -m "$3") BUCKET=$4 T=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$PRE/rgb" "$PRE/depth" "$PRE/data"
for p in "" _depth; do  # this shard's rows, next to the dataset so relative media paths resolve
  awk -v k="$K" -v t="$T" '(NR-1)%k==t' "$DS/dataset$p.jsonl" > "$DS/.part${p}_$T.jsonl"
done
ln -sfn "$PRE/rgb/latents" "$PRE/depth/latents"; ln -sfn "$PRE/rgb/conditions" "$PRE/depth/conditions"
ln -sfn "$PRE/rgb/latents" "$PRE/data/latents"; ln -sfn "$PRE/rgb/conditions" "$PRE/data/conditions"
ln -sfn "$PRE/rgb/reference_latents" "$PRE/data/ref_rgb"; ln -sfn "$PRE/depth/reference_latents" "$PRE/data/ref_depth"
cd "$DEPS/LTX-2/packages/ltx-trainer"
unset PYTHONPATH
for p in "" _depth; do
  out=$PRE/rgb; [ -n "$p" ] && out=$PRE/depth
  "$DEPS/LTX-2/.venv/bin/python" scripts/process_dataset.py "$DS/.part${p}_$T.jsonl" --resolution-buckets "$BUCKET" \
    --model-path "$REPO/models/ltx-2.3/ltx-2.3-22b-dev.safetensors" \
    --text-encoder-path "$REPO/models/gemma-3-12b-it-qat-q4_0-unquantized" \
    --reference-downscale-factor 2 --reshape-mode stretch --fps 15 --skip-audio --output-dir "$out" 2>&1 \
    | { grep -v 'leaked\|nanobind' || true; } | tail -5
done
