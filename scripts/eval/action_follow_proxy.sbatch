#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%A_%a.log
# Counterfactual proxies (simpwm/droid/run_counterfactual.py run COND) of shard TASK of N of an episode list, 3 processes
# sharing one GPU, with the test proxies' settings (per-episode test cameras, stereo depth, 1280x704 control videos,
# the 241-step clip only). Finished episodes are skipped. Submit from the repository root.
# usage: sbatch -a 0-9 scripts/eval/action_follow_proxy.sbatch COND IDS_TXT N OUT_RUN ACTIONS_DIR
# env:   STEREO_DIR first-frame stereo depth (default runs/stereo/depth)
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
COND=$1 IDS=$2 N=$3 OUT=$4 ACT=$5
STEREO_DIR=${STEREO_DIR:-$RUNS/stereo/depth}
T=${SLURM_ARRAY_TASK_ID:-0}
read -ra ids <<< "$(shard_ids "$IDS" "$T" "$N")"
n=${#ids[@]}
echo "task $T: $n episodes"
for j in 0 1 2; do
  part=("${ids[@]:$((j * n / 3)):$(((j + 1) * n / 3 - j * n / 3))}")
  [ ${#part[@]} -gt 0 ] || continue
  "$PY" -W ignore -m simpwm.droid.run_counterfactual run "$COND" "$OUT" "${part[@]}" --actions-dir "$ACT" \
    --camera-dir "$REPO/resources/droid/cameras" --stereo-dir "$STEREO_DIR" --ref-size 1280x704 \
    --clip-frames 241 --clip-only > "$REPO/logs/proxy-$COND-$T-$j.log" 2>&1 &
done
wait
