#!/bin/bash
# GR1 LTX datasets (link mode, simpwm.video.dataset_gr1) from the proxies, and the IC-LoRA training config:
# RUN/ds_train (the first N_PICK candidates above the arm-IoU threshold, in candidate order), RUN/ds_val (the next 4:
# validation samples during training, never trained on), RUN/ds_eval (the 100 test windows), RUN/train/cfg.yaml
# (simpwm.video.make_config --preset gr1, data root RUN/latents/data of scripts/gr1/latents.sbatch).
# usage: scripts/gr1/build_training_data.sh [RUN=runs/gr1] [N_PICK=3000]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
RUN=$(realpath -m "${1:-$RUNS/gr1}"); NPICK=${2:-3000}
"$PY" -m simpwm.video.dataset_gr1 train "$RUN/proxies" "$RUN/ds_train" --items "$RUN/candidates.json" --max-items "$NPICK"
"$PY" -m simpwm.video.dataset_gr1 val "$RUN/proxies" "$RUN/ds_val" --items "$RUN/candidates.json" --exclude "$RUN/ds_train"
"$PY" -m simpwm.video.dataset_gr1 eval "$RUN/proxies_eval" "$RUN/ds_eval" --items "$REPO/resources/gr1/test_windows.json"
read -r -a VAL < <("$PY" -c "import json, sys; print(' '.join(str(json.loads(l)['idx']) for l in open(sys.argv[1])))" \
  "$RUN/ds_val/dataset.jsonl")
mkdir -p "$RUN/train"
"$PY" -m simpwm.video.make_config "$RUN/latents/data" "$RUN/ds_val" "$RUN/train" "$RUN/train/cfg.yaml" "${VAL[@]}" --preset gr1
