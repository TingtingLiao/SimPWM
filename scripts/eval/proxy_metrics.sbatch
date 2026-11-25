#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%A_%a.log
# Clip-level proxy metrics (simpwm/eval/proxy_metrics.py run) for shard TASK, one JSONL per shard.
# The arguments are those of `python -m simpwm.eval.proxy_metrics`; the word TASK in them becomes the array task id. Submit
# from the repository root.
# usage: sbatch -a 0-9 scripts/eval/proxy_metrics.sbatch run OUT/metrics/NAME_TASK.jsonl RUN REAL_DYN --ids IDS_TXT
#          --shard TASK 10
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
args=()
for x in "$@"; do args+=("${x//TASK/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" -W ignore -m simpwm.eval.proxy_metrics "${args[@]}"
