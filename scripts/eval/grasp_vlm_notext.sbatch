#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%A_%a.log
# No-task-text analysis (simpwm/eval/grasp_vlm_notext.py); the VLM decides the same-object questions.
# The arguments are those of `python -m simpwm.eval.grasp_vlm_notext`; the word TASK in them becomes the array task id. Submit
# from the repository root.
# usage: sbatch scripts/eval/grasp_vlm_notext.sbatch OUT_JSON --ids IDS_TXT --with-text ROOT --no-text ROOT --proxy DIR
#          --test-set DS
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
args=()
for x in "$@"; do args+=("${x//TASK/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" -W ignore -m simpwm.eval.grasp_vlm_notext "${args[@]}"
