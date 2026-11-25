#!/bin/bash
# Table 2 (real-to-sim proxy ablation, 97-step protocol) on the DROID test set, step by step (docs/evaluation.md). Each
# step waits for its Slurm arrays: start it under nohup / tmux on a login node. Needs the proxy runs of the rows, the
# test dataset and Table 3's real-video VLM answers (video_ablation.sh); PointWorld row only with PW_ROLLOUTS / PW_VIDEOS.
# usage: scripts/eval/proxy_ablation.sh [OUT=runs/table2]
# env:   ROWS="full=RUN NAME=RUN ..." (default: runs/droid/proxies and runs/droid/ablations/{cam,mono,snap,sam3}),
#        PW_ROLLOUTS, PW_VIDEOS, T3 (Table 3 output, default runs/table3)
set -euo pipefail
source "$(dirname "$0")/../env.sh"
cd "$REPO"
R=$RUNS
OUT=${1:-$R/table2}
T3=${T3:-$R/table3}
TEST=resources/droid/splits/test.txt
DS=$R/droid/datasets/test
A=$R/droid/ablations
read -ra rows <<< "${ROWS:-full=$R/droid/proxies cam=$A/cam mono=$A/mono snap=$A/snap sam3=$A/sam3}"
FULL=${rows[0]#full=}
names=()
row_args=()
for r in "${rows[@]}"; do names+=("${r%%=*}"); row_args+=(--row "$r"); done
pw_args=()
if [ -n "${PW_ROLLOUTS:-}" ]; then pw_args=(--pw-rollouts "$PW_ROLLOUTS" --pw-videos "${PW_VIDEOS:?set PW_VIDEOS with PW_ROLLOUTS}"); names+=(pw); fi
T2=(--ids $TEST --real-dyn "$OUT/real" "${row_args[@]}" ${pw_args[@]+"${pw_args[@]}"})
mkdir -p "$OUT/metrics" logs

# 1. real-video masks shared by the rows: SAM 3 robot instances, grasp targets tracked from the full system's camera
sbatch --wait -J t2-real -a 0-23 scripts/eval/real_dyn.sbatch "$OUT/real" "$FULL" --ids $TEST --shard TASK 24

# 2. per row: arm IoU, RGB L1 / L2, depth L2 (proxy_metrics); automatic grasp success and object IoU against the real
#    targets of the full system's camera (grasp_iou --lenient); VLM grasp answers on the RGB proxy video
for r in "${rows[@]}"; do
  name=${r%%=*} run=${r#*=}
  sbatch --wait -J t2-metrics -a 0-9 scripts/eval/proxy_metrics.sbatch run "$OUT/metrics/${name}_TASK.jsonl" "$run" \
    "$OUT/real" --ids $TEST --shard TASK 10 &
  sbatch --wait -J t2-grasp -a 0-19 scripts/eval/grasp_iou.sbatch "$run" --ids $TEST --shard TASK 20 --clip-frames 97 \
    --lenient --out-name grasp97.json --gt-run "$FULL" &
  sbatch --wait -J t2-vlm -a 0-4 scripts/eval/grasp_vlm.sbatch answer "$OUT/vlm" "$name" \
    "$run/droid_{i5}/conditions/ref_rgb_1280x704.mp4" --ids $TEST --shard TASK 5 &
done
# 3. image object IoU of every row; PointWorld's RGB / depth metrics; every row on common pixels (CPU)
for name in "${names[@]}"; do
  sbatch --wait -J t2-obj -a 0-19 scripts/eval/grasp_vlm_ablation.sbatch object-iou "$name" "$OUT" "${T2[@]}" \
    --shard TASK 20 &
done
if [ ${#pw_args[@]} -gt 0 ]; then
  sbatch --wait -J t2-vlm -a 0-4 scripts/eval/grasp_vlm.sbatch answer "$OUT/vlm" pw "$PW_VIDEOS" --ids $TEST \
    --shard TASK 5 &
  sbatch --wait -J t2-pw -a 0-19 scripts/eval/grasp_vlm_ablation.sbatch pw-metrics "$OUT" "${T2[@]}" --shard TASK 20 &
fi
sbatch --wait -J t2-aligned -a 0-24 scripts/eval/grasp_vlm_ablation.sbatch aligned "$OUT" "${T2[@]}" --shard TASK 25 &
wait_all

# 4. VLM grasp decisions against the real-video answers, then the summaries
sbatch --wait -J t2-match scripts/eval/grasp_vlm.sbatch match "$OUT/vlm" "${names[@]}" --gt "$T3/vlm/real" \
  --test-set "$DS"
"$PY" -m simpwm.eval.proxy_metrics summary "$OUT/auto.json" "$OUT/metrics" --ids $TEST "${row_args[@]}" \
  --grasp-name grasp97.json
"$PY" -m simpwm.eval.grasp_vlm_ablation summary "$OUT" "${T2[@]}" --auto "$OUT/auto.json" | tee "$OUT/summary.txt"
