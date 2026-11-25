#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.log
# The "moved" rule (simpwm/eval/action_follow.py moved) for several video sets, shard TASK of N of ROOT/ids100.txt.
# SPEC = NAME:COND:GEN_DIR:STRIDE (GEN_DIR "real": the recordings). Submit from the repository root.
# usage: sbatch -a 0-9 scripts/eval/action_follow_moved.sbatch ROOT REF_RUN N SPEC [SPEC ...]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
ROOT=$1 REF=$2 N=$3
shift 3
for spec in "$@"; do
  IFS=: read -r name cond gen stride <<< "$spec"
  "$PY" -W ignore -m simpwm.eval.action_follow moved "$name" "$cond" "$gen" --stride "$stride" --root "$ROOT" --ref-run "$REF" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" "$N"
done
