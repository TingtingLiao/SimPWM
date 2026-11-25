#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%A_%a.log
# Grasp agreement and grasped-object IoU of a proxy run (simpwm/eval/grasp_iou.py) for shard TASK; finished episodes
# are skipped.
# The arguments are those of `python -m simpwm.eval.grasp_iou`; the word TASK in them becomes the array task id. Submit
# from the repository root.
# usage: sbatch -a 0-19 scripts/eval/grasp_iou.sbatch RUN --ids IDS_TXT --shard TASK 20 [--clip-frames 97 --lenient ...]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
args=()
for x in "$@"; do args+=("${x//TASK/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" -W ignore -m simpwm.eval.grasp_iou "${args[@]}"
