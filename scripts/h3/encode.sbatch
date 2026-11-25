#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.log
# H3 encodings of shard TASK of a paired dataset (simpwm/video/dataset.py --link): VAE latents of the proxy, the first
# frame and the real video, then the Qwen3-VL conditioning. Existing files are skipped, so shards share CACHE and can
# be resubmitted. Further arguments go to simpwm.h3.preencode (--ids FILE, --no-target for test sets).
# Submit from the repository root.
# usage: sbatch -a 0-23 scripts/h3/encode.sbatch DS_DIR CACHE 24 [--ids resources/droid/splits/train_h3.txt]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
source "$REPO/scripts/h3/env.sh"
DS=$1 CACHE=$2 K=$3 T=${SLURM_ARRAY_TASK_ID:-0}
shift 3
for st in vae text; do
  "$H3_PY" -m simpwm.h3.preencode --stage $st --data "$DS" --cache "$CACHE" --shard "$T" --nshards "$K" "$@"
done
