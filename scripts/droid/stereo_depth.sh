#!/bin/bash
# First-frame ext1 stereo depth of an episode list -> runs/stereo/depth/<idx>_ext1.npy: download the ext1 .svo
# recordings, then the three extraction stages as Slurm arrays (each step waits for the previous one; already
# processed episodes are skipped). Long-running: start it under nohup / tmux on a login node.
# usage: scripts/droid/stereo_depth.sh IDS_TXT
# env:   ZED_HOME, FS_PY, FS_SHIM (see scripts/droid/stereo.sbatch and docs/dataset.md)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
cd "$REPO"
"$PY" -m simpwm.droid.stereo.download --episodes "$1" --svo ext1 | tail -2
for stage in rlds svo fs; do
  sbatch --wait -J "stereo-$stage" -a 0-15 scripts/droid/stereo.sbatch "$stage" 16
done
