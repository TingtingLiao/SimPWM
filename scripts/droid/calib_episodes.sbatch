#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=90G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.log
# Per-episode camera fits (simpwm/droid/episode_calib.py) of shard TASK of an episode list, 3 processes sharing one GPU.
# Finished episodes are skipped, so a shard can be resubmitted. Extra arguments go to simpwm.droid.episode_calib (the
# paper's cameras: --clip-frames 241). Submit from the repository root.
# usage: sbatch -a 0-47 scripts/droid/calib_episodes.sbatch IDS_TXT N_TASKS OUT_DIR [--clip-frames 241]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
IDS=$1 N=$2 OUT=$3
shift 3
T=${SLURM_ARRAY_TASK_ID:-0}
read -ra ids <<< "$(shard_ids "$IDS" "$T" "$N")"
n=${#ids[@]}
echo "task $T: $n episodes"
for j in 0 1 2; do
  part=("${ids[@]:$((j * n / 3)):$(((j + 1) * n / 3 - j * n / 3))}")
  [ ${#part[@]} -gt 0 ] || continue
  "$PY" -W ignore -m simpwm.droid.episode_calib "$OUT" "${part[@]}" "$@" \
    > "$REPO/logs/cameras-$(basename "$OUT")-$T-$j.log" 2>&1 &
done
wait
