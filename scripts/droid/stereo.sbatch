#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%A_%a.log
# First-frame ext1 stereo depth of the downloaded recordings (simpwm/droid/stereo/extract.py), one shard per array task.
# Stages, in order: rlds (main env, first frames), svo (ZED SDK venv: rectified pairs), fs (FoundationStereo venv:
# depth -> runs/stereo/depth/<idx>_ext1.npy). Submit from the repository root.
# usage: sbatch -a 0-15 scripts/droid/stereo.sbatch {rlds|svo|fs} 16
# env:   ZED_HOME (svo: ex/ SDK, deps/lib/ libturbojpeg, venv/ pyzed, settings/), FS_PY + FS_SHIM (fs); docs/dataset.md
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}/scripts/env.sh"
STAGE=$1 N=$2
ARGS=(--task "${SLURM_ARRAY_TASK_ID:-0}" --n-tasks "$N")
case $STAGE in
  rlds)
    "$PY" -W ignore -m simpwm.droid.stereo.extract rlds --cams ext1 "${ARGS[@]}" ;;
  svo)
    : "${ZED_HOME:?set ZED_HOME (see docs/dataset.md)}"
    export LD_LIBRARY_PATH=$ZED_HOME/ex/lib:$ZED_HOME/deps/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} PYTHONPATH=$REPO
    "$ZED_HOME/venv/bin/python" -W ignore -m simpwm.droid.stereo.extract svo --cams ext1 \
      --zed-settings "$ZED_HOME/settings" "${ARGS[@]}" ;;
  fs)
    : "${FS_PY:?set FS_PY (see docs/dataset.md)}" "${FS_SHIM:?set FS_SHIM (see docs/dataset.md)}"
    export PYTHONPATH=$FS_SHIM:$DEPS/FoundationStereo:$REPO
    cd "$DEPS/FoundationStereo"
    "$FS_PY" -W ignore -m simpwm.droid.stereo.extract fs "${ARGS[@]}" ;;
  *)
    echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
