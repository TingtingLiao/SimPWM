#!/bin/bash
# The DROID test set (resources/droid/splits/test.txt: 500 episodes from setups without training episodes) with its
# shipped per-episode cameras (resources/droid/cameras): stereo depth, 241-frame proxies and the link-mode
# evaluation dataset that scripts/droid/generate.sbatch renders (docs/dataset.md). Each step waits for its Slurm array:
# start it under nohup / tmux on a login node.
# usage: scripts/droid/build_test_set.sh [OUT_DIR=runs/droid]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
cd "$REPO"
D=${1:-$RUNS/droid}
TEST=resources/droid/splits/test.txt
mkdir -p "$D"

# 1. first-frame stereo depth -> runs/stereo/depth
scripts/droid/stereo_depth.sh $TEST
# 2. proxies of the first 241 recorded steps -> D/proxies
sbatch --wait -J proxies-test -a 0-39 scripts/droid/proxy.sbatch $TEST 40 "$D/proxies" \
  --camera-dir resources/droid/cameras --clip-frames 241 --clip-only
# 3. link-mode evaluation dataset (+ the robot-only reference of the proxy ablation) -> D/datasets/test
"$PY" -m simpwm.video.dataset "$D/proxies" "$D/datasets/test" @$TEST --link --size 1280x704 --clip-frames 241 | tail -1
"$PY" -m simpwm.video.robot_refs "$D/proxies" "$D/datasets/test" --clip-frames 97
echo "test set ready: $D/datasets/test"
