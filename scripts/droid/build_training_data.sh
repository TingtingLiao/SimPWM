#!/bin/bash
# Training data of the DROID video models from the paper's split (resources/droid/splits/{train,val}.txt) and its
# shipped per-episode cameras (resources/droid/cameras): stereo depth, 241-frame proxies, link-mode datasets,
# 241-frame latents and the 97-frame latents sliced from them (docs/train.md). Each step waits for its Slurm array: start it under nohup / tmux on a login node.
# usage: scripts/droid/build_training_data.sh [OUT_DIR=runs/droid]
# env:   as scripts/env.sh, plus ZED_HOME, FS_PY, FS_SHIM for the stereo step
set -euo pipefail
source "$(dirname "$0")/../env.sh"
cd "$REPO"
D=${1:-$RUNS/droid}
SPLITS=resources/droid/splits
mkdir -p "$D"
cat $SPLITS/val.txt $SPLITS/train.txt > "$D/ids_train_val.txt"

# 1. first-frame stereo depth -> runs/stereo/depth
scripts/droid/stereo_depth.sh "$D/ids_train_val.txt"
# 2. proxies of the first 241 recorded steps, without the evaluation against the real video -> D/proxies
sbatch --wait -J proxies -a 0-31 scripts/droid/proxy.sbatch "$D/ids_train_val.txt" 32 "$D/proxies" \
  --camera-dir resources/droid/cameras --clip-frames 241 --clip-only
# 3. link-mode datasets (raw ext1 MP4 + RGB / depth proxies at 1280x704) -> D/datasets/{train,val}
for s in train val; do
  "$PY" -m simpwm.video.dataset "$D/proxies" "$D/datasets/$s" @$SPLITS/$s.txt --link --size 1280x704 --clip-frames 241 \
    | tail -1
done
# 4. 241-frame latents, then the 97-frame latents sliced from them (the video VAE is causal)
sbatch --wait -J latents -a 0-31 scripts/droid/latents.sbatch "$D/datasets/train" 32 "$D/latents241" 1280x704x241
"$PY" -m simpwm.video.slice_latents "$D/latents241/data" "$D/latents97/data" latents ref_rgb ref_depth --clip-frames 97
# 5. robot-only reference (proxy ablation) on top of the 97-frame latents -> D/latents97_robot
for s in train val; do
  "$PY" -m simpwm.video.robot_refs "$D/proxies" "$D/datasets/$s" --clip-frames 97
done
sbatch --wait -J latents-robot -a 0-31 scripts/droid/latents_ref.sbatch "$D/datasets/train" robot 32 \
  "$D/latents97/data" "$D/latents97_robot" 1280x704x97
echo "training data ready: $D/latents97/data, $D/latents97_robot/data, $D/latents241/data"
