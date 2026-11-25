#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=120G
#SBATCH -t 04:00:00
# GR1 proxies + real clips (simpwm.gr1.run), NPROC processes on one GPU per array task, with the paper's hand scale;
# items already done are skipped, so tasks can be resubmitted. MODE train: RUN/candidates.json (simpwm.gr1.select) ->
# RUN/proxies; MODE eval: the DreamDojo test windows -> RUN/proxies_eval. Extra arguments go to simpwm.gr1.run.
# usage (from the repo root): sbatch -a 0-K-1 scripts/gr1/proxy.sbatch MODE K [RUN=runs/gr1] [NPROC=3] [-- ARGS]
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"  # sbatch runs a copy of this file
MODE=$1; K=$2; shift 2; RUN=$RUNS/gr1; NP=3; T=${SLURM_ARRAY_TASK_ID:-0}
if [ $# -gt 0 ] && [ "$1" != "--" ]; then RUN=$1; shift; fi
if [ $# -gt 0 ] && [ "$1" != "--" ]; then NP=$1; shift; fi
if [ $# -gt 0 ]; then shift; fi  # "--"
if [ "$MODE" = train ]; then OUT=$RUN/proxies; ITEMS=(--items "$RUN/candidates.json"); else OUT=$RUN/proxies_eval; ITEMS=(); fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
for j in $(seq 0 $((NP - 1))); do
  "$PY" -W ignore -m simpwm.gr1.run "$MODE" "$OUT" ${ITEMS[@]+"${ITEMS[@]}"} --shard $((T * NP + j)) --num-shards $((K * NP)) \
    --hand-scale 0.89:1.28 "$@" > "$REPO/logs/gr1-proxy-$MODE-$T-$j-${SLURM_JOB_ID:-local}.log" 2>&1 &
done
wait
