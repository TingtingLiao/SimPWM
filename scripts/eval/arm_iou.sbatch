#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.log
# Arm + gripper IoU of a proxy run (simpwm/eval/arm_iou.py) for shard TASK; finished episodes are skipped.
# The arguments are those of `python -m simpwm.eval.arm_iou`; the word TASK in them becomes the array task id. Submit
# from the repository root.
# usage: sbatch -a 0-31 scripts/eval/arm_iou.sbatch RUN --ids IDS_TXT --shard TASK 32 [--clip-frames 241]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
args=()
for x in "$@"; do args+=("${x//TASK/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" -W ignore -m simpwm.eval.arm_iou "${args[@]}"
