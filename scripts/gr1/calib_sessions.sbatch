#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 04:00:00
# Per-session GR1 head-camera refit (simpwm.gr1.session_calib), NPROC processes on one GPU per array task, each a shard of
# resources/gr1/sessions.json; sessions already fitted are skipped, so tasks can be resubmitted.
# usage (from the repo root): sbatch -a 0-K-1 scripts/gr1/calib_sessions.sbatch K [OUT=runs/gr1/session_cameras] [NPROC=2]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"  # sbatch runs a copy of this file
K=$1; OUT=${2:-$RUNS/gr1/session_cameras}; NP=${3:-2}; T=${SLURM_ARRAY_TASK_ID:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
for j in $(seq 0 $((NP - 1))); do
  "$PY" -W ignore -m simpwm.gr1.session_calib --out "$OUT" --shard $((T * NP + j)) --num-shards $((K * NP)) \
    > "$REPO/logs/gr1-calib-$T-$j-${SLURM_JOB_ID:-local}.log" 2>&1 &
done
wait
