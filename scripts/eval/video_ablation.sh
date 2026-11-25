#!/bin/bash
# Table 3 (sim-to-real ablation of the video model) on the DROID test set, step by step (docs/evaluation.md): the test
# videos of every model, the world-model metrics at the models' native output (wm_bench --protocol table3) and the VLM
# grasp judge against the real videos. Each step waits for its Slurm arrays: start it under nohup / tmux.
# usage: scripts/eval/video_ablation.sh [OUT=runs/table3] [MODEL ...]   (default: noproxy robot pw rgb depth rgb_depth)
# env:   TRAIN (training output: <model>.yaml, <model>/checkpoints; default runs/droid/train), STEP (default 2000)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
cd "$REPO"
R=$RUNS
OUT=${1:-$R/table3}
[ $# -gt 0 ] && shift
models=("$@")
[ ${#models[@]} -gt 0 ] || models=(noproxy robot pw rgb depth rgb_depth)
TRAIN=${TRAIN:-$R/droid/train}
STEP=$(printf %05d "${STEP:-2000}")
TEST=resources/droid/splits/test.txt
DS=$R/droid/datasets/test
REF=$R/droid/proxies
BENCH=(--ids $TEST --test-set "$DS" --protocol table3)
mkdir -p logs

# 1. test videos of every model; meanwhile the real-side benchmark caches and the VLM answers on the real videos (GT)
for m in "${models[@]}"; do
  sbatch --wait -J t3-gen -a 0-19 scripts/droid/generate.sbatch "$TRAIN/$m.yaml" \
    "$TRAIN/$m/checkpoints/lora_weights_step_$STEP.safetensors" "$DS" "$OUT/generations/$m" 20 &
done
sbatch --wait -J t3-real -a 0-24 scripts/eval/wm_bench.sbatch real "$OUT/bench" "${BENCH[@]}" --ref-run "$REF" \
  --shard TASK 25 &
sbatch --wait -J t3-vlm -a 0-4 scripts/eval/grasp_vlm.sbatch answer "$OUT/vlm" real "$DS/videos" --ids $TEST \
  --shard TASK 5 &
wait_all

# 2. per model: the generated half of the panels -> videos/<model>/<idx>.mp4, metrics, Interact., VLM grasp answers
for m in "${models[@]}"; do
  "$PY" -m simpwm.eval.wm_bench export "$OUT/generations/$m" "$OUT/videos/$m"
  sbatch --wait -J t3-bench -a 0-24 scripts/eval/wm_bench.sbatch gen "$OUT/bench" "$m" "$OUT/videos/$m" "${BENCH[@]}" \
    --ref-run "$REF" --shard TASK 25 &
  sbatch --wait -J t3-interact -a 0-4 scripts/eval/wm_bench.sbatch vlm "$OUT/bench" "$m" "$OUT/videos/$m" \
    "${BENCH[@]}" --shard TASK 5 &
  sbatch --wait -J t3-vlm -a 0-4 scripts/eval/grasp_vlm.sbatch answer "$OUT/vlm" "$m" "$OUT/videos/$m" --ids $TEST \
    --shard TASK 5 &
done
wait_all

# 3. VLM grasp decisions against the real videos, then the summaries
sbatch --wait -J t3-match scripts/eval/grasp_vlm.sbatch match "$OUT/vlm" "${models[@]}" --test-set "$DS"
"$PY" -m simpwm.eval.wm_bench summary "$OUT/bench" "$OUT/bench_summary.json" "${models[@]}" --protocol table3 > /dev/null
"$PY" -m simpwm.eval.grasp_vlm summary "$OUT/vlm" "${models[@]}" > /dev/null
echo "Table 3: $OUT/bench_summary.json, $OUT/vlm/summary.json"
