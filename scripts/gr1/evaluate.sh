#!/bin/bash
# GR1 comparison with DreamDojo on its 100 test windows (simpwm.eval.gr1_bench, one GPU): RUN/bench.json (PSNR, SSIM,
# LPIPS, FVD, robot IoU of both models) and comparison grids in RUN/grids.
# usage: scripts/gr1/evaluate.sh [RUN=runs/gr1]   (needs RUN/gen, RUN/dreamdojo, RUN/proxies_eval)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
RUN=${1:-$RUNS/gr1}
DIRS=(--gen "$RUN/gen" --dreamdojo "$RUN/dreamdojo" --proxies "$RUN/proxies_eval")
"$PY" -W ignore -m simpwm.eval.gr1_bench score "${DIRS[@]}" --out "$RUN/bench.json" --figs "$RUN/bench"
"$PY" -W ignore -m simpwm.eval.gr1_bench grids "${DIRS[@]}" --score "$RUN/bench.json" --out "$RUN/grids"
