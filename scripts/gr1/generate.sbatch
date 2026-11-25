#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 08:00:00
# Generate the GR1 test windows with a trained IC-LoRA through the trainer's own validation pipeline (config:
# simpwm.video.eval_config, 49 frames): one GPU per shard; samples OUT/shard_T/samples/step_000000_<k>.mp4 = the k-th window of
# OUT/shard_T/order.json, as [reference | generation] panels. Finished shards are skipped.
# usage (from the repo root): sbatch -a 0-N-1 scripts/gr1/generate.sbatch TRAIN_CFG LORA EVAL_DS OUT N
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"  # sbatch runs a copy of this file
CFG=$1; LORA=$2; EV=$3; OUT=$(realpath -m "$4"); N=$5; T=${SLURM_ARRAY_TASK_ID:-0}
LTX=$DEPS/LTX-2
S=$OUT/shard_$T
"$PY" -m simpwm.video.eval_config "$CFG" "$LORA" "$EV" "$S" --shard "$T" "$N" --clip-frames 49
n=$("$PY" -c "import json, sys; print(len(json.load(open(sys.argv[1]))))" "$S/order.json")
if [ "$(find "$S" -path '*/samples/step_000000_*.mp4' | wc -l)" -ge "$n" ]; then echo "shard $T done"; exit 0; fi
export SIMPWM_REF_POS_FIX=1
cd "$LTX/packages/ltx-trainer"; unset PYTHONPATH
"$LTX/.venv/bin/accelerate" launch --config_file "$REPO/configs/accelerate/single_gpu.yaml" scripts/train.py "$S/cfg.yaml"
