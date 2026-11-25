#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.log
# VLM grasp judge (simpwm/eval/grasp_vlm.py): answers of a video set for shard TASK, or the match stage.
# The arguments are those of `python -m simpwm.eval.grasp_vlm`; the word TASK in them becomes the array task id. Submit
# from the repository root.
# usage: sbatch -a 0-4 scripts/eval/grasp_vlm.sbatch answer ROOT NAME VIDEOS --ids IDS_TXT --shard TASK 5 [--stride 3]
#        sbatch scripts/eval/grasp_vlm.sbatch match ROOT NAME [NAME ...] --test-set DS
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
args=()
for x in "$@"; do args+=("${x//TASK/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" -W ignore -m simpwm.eval.grasp_vlm "${args[@]}"
