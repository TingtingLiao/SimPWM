#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%A_%a.log
# One more reference stream for an existing preprocessed data root (as latents.sbatch's depth pass), shard TASK:
# DS/dataset_KIND.jsonl -> PRE/KIND/reference_latents, with PRE/KIND/{latents,conditions} linking DATA's so only the
# reference is encoded; PRE/data/{latents,conditions} link DATA's and PRE/data/ref_KIND the new reference latents.
# Example: KIND=robot (simpwm/video/robot_refs.py), DATA = the 97-frame data root. Submit from the repository root.
# usage: sbatch -a 0-31 scripts/droid/latents_ref.sbatch DS_DIR KIND 32 DATA PRE WxHxF   (e.g. 1280x704x97)
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
DS=$(realpath "$1") KIND=$2 K=$3 DATA=$(realpath "$4") PRE=$(realpath -m "$5") BUCKET=$6 T=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$PRE/$KIND" "$PRE/data"
awk -v k="$K" -v t="$T" '(NR-1)%k==t' "$DS/dataset_$KIND.jsonl" > "$DS/.part_${KIND}_$T.jsonl"
ln -sfn "$DATA/latents" "$PRE/$KIND/latents"; ln -sfn "$DATA/conditions" "$PRE/$KIND/conditions"
ln -sfn "$DATA/latents" "$PRE/data/latents"; ln -sfn "$DATA/conditions" "$PRE/data/conditions"
ln -sfn "$PRE/$KIND/reference_latents" "$PRE/data/ref_$KIND"
cd "$DEPS/LTX-2/packages/ltx-trainer"
unset PYTHONPATH
"$DEPS/LTX-2/.venv/bin/python" scripts/process_dataset.py "$DS/.part_${KIND}_$T.jsonl" --resolution-buckets "$BUCKET" \
  --model-path "$REPO/models/ltx-2.3/ltx-2.3-22b-dev.safetensors" \
  --text-encoder-path "$REPO/models/gemma-3-12b-it-qat-q4_0-unquantized" \
  --reference-downscale-factor 2 --reshape-mode stretch --fps 15 --skip-audio --output-dir "$PRE/$KIND" 2>&1 \
  | { grep -v 'leaked\|nanobind' || true; } | tail -5
