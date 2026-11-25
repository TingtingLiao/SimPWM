#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=160G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.log
# Generate the videos of an evaluation dataset (simpwm/video/dataset.py output) with a trained IC-LoRA through the
# trainer's own validation pipeline, so inference matches training-time validation: shard TASK of N, one GPU. The
# reference streams are those of the training config's validation samples. Samples land in
# OUT/shard_T/samples/step_000000_<k>.mp4 (k-th episode of OUT/shard_T/order.json); finished shards are skipped.
# Submit from the repository root.
# usage: sbatch -a 0-19 scripts/droid/generate.sbatch TRAIN_CONFIG LORA EVAL_DIR OUT 20
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
CFG=$(realpath "$1") LORA=$(realpath "$2") EV=$(realpath "$3") OUT=$(realpath -m "$4") N=$5 T=${SLURM_ARRAY_TASK_ID:-0}
S=$OUT/shard_$T
mkdir -p "$S"
"$PY" -m simpwm.video.eval_config "$CFG" "$LORA" "$EV" "$S" --shard "$T" "$N"
n=$("$PY" -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" "$S/order.json")
if [ "$(find "$S/samples" -name 'step_000000_*.mp4' 2>/dev/null | wc -l)" -ge "$n" ]; then
  echo "shard $T done"
  exit 0
fi
cd "$DEPS/LTX-2/packages/ltx-trainer"
unset PYTHONPATH
"$DEPS/LTX-2/.venv/bin/accelerate" launch --config_file "$REPO/configs/accelerate/single_gpu.yaml" scripts/train.py \
  "$S/cfg.yaml" 2>&1 | { grep --line-buffered -v 'leaked\|nanobind' || true; }
