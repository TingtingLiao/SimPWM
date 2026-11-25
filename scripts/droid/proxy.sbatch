#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%A_%a.log
# Proxies + 1280x704 LTX control videos (simpwm/droid/run.py) of shard TASK of an episode list, 3 processes sharing one
# GPU. Finished episodes are skipped, so a shard can be resubmitted. Extra arguments go to simpwm.droid.run (e.g.
# --camera-dir DIR --clip-frames 241 --clip-only). Submit from the repository root.
# usage: sbatch -a 0-31 scripts/droid/proxy.sbatch IDS_TXT N_TASKS OUT_DIR [RUN_OPTIONS ...]
# env:   STEREO_DIR first-frame stereo depth (default runs/stereo/depth; set empty for monocular depth)
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
IDS=$1 N=$2 OUT=$3
shift 3
STEREO_DIR=${STEREO_DIR-$RUNS/stereo/depth}
T=${SLURM_ARRAY_TASK_ID:-0}
read -ra ids <<< "$(shard_ids "$IDS" "$T" "$N")"
n=${#ids[@]}
echo "task $T: $n episodes"
for j in 0 1 2; do
  part=("${ids[@]:$((j * n / 3)):$(((j + 1) * n / 3 - j * n / 3))}")
  [ ${#part[@]} -gt 0 ] || continue
  "$PY" -W ignore -m simpwm.droid.run "$OUT" "${part[@]}" ${STEREO_DIR:+--stereo-dir "$STEREO_DIR"} --ref-size 1280x704 "$@" \
    > "$REPO/logs/proxy-$(basename "$OUT")-$T-$j.log" 2>&1 &
done
wait
