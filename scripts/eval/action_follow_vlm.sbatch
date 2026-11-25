#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.log
# The VLM judge (simpwm/eval/action_follow.py vlm) for several video sets, shard TASK of N of ROOT/ids100.txt.
# SPEC = NAME:COND:GEN_DIR:STRIDE (GEN_DIR "real": the recordings). Submit from the repository root.
# usage: sbatch -a 0-9 scripts/eval/action_follow_vlm.sbatch ROOT N SPEC [SPEC ...]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
ROOT=$1 N=$2
shift 2
for spec in "$@"; do
  IFS=: read -r name cond gen stride <<< "$spec"
  "$PY" -W ignore -m simpwm.eval.action_follow vlm "$name" "$cond" "$gen" --stride "$stride" --root "$ROOT" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" "$N"
done
