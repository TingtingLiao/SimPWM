#!/bin/bash
# Render the proxies of `python -m simpwm.demo` with SimPWM-H3: encode every example of OUT/dataset.jsonl, then
# generate OUT/h3c/<name>.mp4 (158 frames, 24 fps, 1280x704: the first 97 steps) and <name>_15fps.mp4. Needs the H3
# environment, the converted base model and the LoRA (docs/installation.md, docs/inference.md), and 4 GPUs of 80 GB (about 27 min per round of 4
# videos on A100). Existing encodings and videos are skipped.
# usage: bash scripts/h3/demo.sh [OUT=runs/demo] [LORA=models/simpwm-h3/simpwm_h3c_1280x704.safetensors]
set -euo pipefail
source "$(dirname "$0")/../env.sh"
source "$REPO/scripts/h3/env.sh"
cd "$REPO"
OUT=${1:-runs/demo} LORA=${2:-models/simpwm-h3/simpwm_h3c_1280x704.safetensors}
for st in vae text; do
  "$H3_PY" -m simpwm.h3.preencode --stage $st --data "$OUT" --cache "$OUT/h3_cache" --no-target --shard 0 --nshards 1
done
NAMES=$("$H3_PY" -c "import json, sys; print(','.join(json.loads(r)['idx'] for r in open(sys.argv[1])))" "$OUT/dataset.jsonl")
torchrun --standalone --nproc-per-node="$(nvidia-smi -L | wc -l)" -m simpwm.h3.sample --cache "$OUT/h3_cache" \
  --out "$OUT" --variant "h3c:$LORA:pin" --names "$NAMES"
