#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.log
# One stage of the action-following evaluation (simpwm/eval/action_follow.py) on a GPU, e.g. targets of shard TASK.
# The arguments are those of `python -m simpwm.eval.action_follow`; the word TASK in them becomes the array task id. Submit
# from the repository root.
# usage: sbatch -a 0-9 scripts/eval/action_follow_eval.sbatch targets --root ROOT --ids IDS_TXT --ref-run RUN --shard TASK 10
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
args=()
for x in "$@"; do args+=("${x//TASK/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" -W ignore -m simpwm.eval.action_follow "${args[@]}"
