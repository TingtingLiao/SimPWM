#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 06:00:00
# DreamDojo baseline on the GR1_robot test split with its own protocol: examples/action_conditioned.py (unmodified,
# through scripts/gr1/run_dreamdojo.py), official 2B GR1 post-train checkpoint, --num-frames 49 --num-samples 100
# --data-split test --deterministic-uniform-sampling (its eval.sh). Shard TASK of K: the windows idx != TASK (mod K)
# get a placeholder <idx>_pred.mp4 in this shard's save dir, which the script's own "already exists" check skips.
# env: DREAMDOJO (DreamDojo checkout, default deps/DreamDojo), DD_PY (its python), DD_CKPT (2B_GR1_post-train dir)
# usage (from the repo root): sbatch -a 0-K-1 scripts/gr1/dreamdojo.sbatch K [OUT=runs/gr1/dreamdojo]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"  # sbatch runs a copy of this file
K=$1; OUT=$(realpath -m "${2:-$RUNS/gr1/dreamdojo}"); T=${SLURM_ARRAY_TASK_ID:-0}
DREAMDOJO=$(realpath "${DREAMDOJO:-$DEPS/DreamDojo}")
DD_PY=${DD_PY:-python}; DD_CKPT=${DD_CKPT:-$DREAMDOJO/checkpoints/2B_GR1_post-train}
GR1=$("$PY" -c "from simpwm.paths import GR1_ROOT; print(GR1_ROOT)")
S=$OUT/shard_$T; mkdir -p "$S/iter_000050000"
for i in $(seq 0 99); do
  if [ $((i % K)) -ne "$T" ]; then touch "$S/iter_000050000/$(printf %04d "$i")_pred.mp4"; fi
done
cd "$DREAMDOJO"; export PYTHONPATH=$DREAMDOJO
"$DD_PY" "$REPO/scripts/gr1/run_dreamdojo.py" --dreamdojo "$DREAMDOJO" -- -o "$S/out" --checkpoints-dir "$DD_CKPT" \
  --experiment dreamdojo_2b_480_640_gr1 --save-dir "$S" --num-frames 49 --num-samples 100 --dataset-path "$GR1" \
  --data-split test --deterministic-uniform-sampling
