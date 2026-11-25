#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%A_%a.log
# Real-video masks shared by the Table 2 rows (simpwm/eval/real_dyn.py) for shard TASK; finished episodes are skipped.
# The arguments are those of `python -m simpwm.eval.real_dyn`; the word TASK in them becomes the array task id. Submit
# from the repository root.
# usage: sbatch -a 0-23 scripts/eval/real_dyn.sbatch OUT REF_RUN --ids IDS_TXT --shard TASK 24
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
args=()
for x in "$@"; do args+=("${x//TASK/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" -W ignore -m simpwm.eval.real_dyn "${args[@]}"
