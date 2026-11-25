#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%A_%a.log
# Which grasp the generations follow when the proxy and the real grasp disagree (simpwm/eval/grasp_vlm_follow.py); the
# VLM decides the same-object questions.
# The arguments are those of `python -m simpwm.eval.grasp_vlm_follow`; the word TASK in them becomes the array task id. Submit
# from the repository root.
# usage: sbatch scripts/eval/grasp_vlm_follow.sbatch OUT_JSON GEN_ROOT --proxy VARIANT=DIR [...] --test-set DS
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
args=()
for x in "$@"; do args+=("${x//TASK/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" -W ignore -m simpwm.eval.grasp_vlm_follow "${args[@]}"
