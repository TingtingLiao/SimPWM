#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%A_%a.log
# One per-episode stage of the Table 2 metrics beyond proxy_metrics (simpwm/eval/grasp_vlm_ablation.py: pw-metrics,
# object-iou ROW, aligned) for shard TASK, on CPUs; finished episodes are skipped.
# The arguments are those of `python -m simpwm.eval.grasp_vlm_ablation`; the word TASK in them becomes the array task id. Submit
# from the repository root.
# usage: sbatch -a 0-19 scripts/eval/grasp_vlm_ablation.sbatch object-iou full OUT --ids IDS_TXT --real-dyn DIR
#          --row full=RUN [--row NAME=RUN ...] [--pw-rollouts DIR --pw-videos DIR] --shard TASK 20
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
args=()
for x in "$@"; do args+=("${x//TASK/${SLURM_ARRAY_TASK_ID:-0}}"); done
"$PY" -W ignore -m simpwm.eval.grasp_vlm_ablation "${args[@]}"
