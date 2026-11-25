#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x-%A_%a.log
# One stage of the world-model benchmark (simpwm/eval/wm_bench.py: real, gen, vlm) for shard TASK; finished episodes
# are skipped.
# The arguments are those of `python -m simpwm.eval.wm_bench`; the word TASK in them becomes the array task id. Submit
# from the repository root.
# usage: sbatch -a 0-24 scripts/eval/wm_bench.sbatch gen OUT NAME GEN_DIR --ids IDS_TXT --ref-run RUN --test-set DS
#          --shard TASK 25 [--protocol table3] [--stride 3]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
args=()
for x in "$@"; do args+=("${x//TASK/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" -W ignore -m simpwm.eval.wm_bench "${args[@]}"
