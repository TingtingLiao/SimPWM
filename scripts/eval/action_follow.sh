#!/bin/bash
# Table 4 (action following) on the DROID test set, step by step (docs/evaluation.md): episode selection, counterfactual
# actions and proxies, the SimPWM videos under each condition, both judges and the summary. The Ctrl-World / DreamDojo rows
# read ROOT/videos/{ctrl_world,dreamdojo}_<cond>/<idx>.mp4, rendered by those models from ROOT/actions. Each step waits
# for its Slurm arrays: start it under nohup / tmux on a login node.
# usage: scripts/eval/action_follow.sh [ROOT=runs/action_follow]
# env:   T3 (Table 3 output holding the SimPWM test videos, default runs/table3), TRAIN (default runs/droid/train), STEP
set -euo pipefail
source "$(dirname "$0")/../env.sh"
cd "$REPO"
R=$RUNS
A=${1:-$R/action_follow}
T3=${T3:-$R/table3}
TRAIN=${TRAIN:-$R/droid/train}
STEP=$(printf %05d "${STEP:-2000}")
TEST=resources/droid/splits/test.txt
REF=$R/droid/proxies
AF=(--root "$A" --ref-run "$REF" --test-set "$R/droid/datasets/test")
mkdir -p "$A" logs

# 1. samples: stalled closures -> strict grasp targets -> the "moved" rule on the candidates' recordings -> 100 episodes
"$PY" -m simpwm.eval.action_follow stalled "${AF[@]}" --test-ids $TEST
sbatch --wait -J af-targets -a 0-9 scripts/eval/action_follow_eval.sbatch targets "${AF[@]}" --ids "$A/ids_stalled.txt" \
  --shard TASK 10
"$PY" -m simpwm.eval.action_follow select "${AF[@]}" --test-ids $TEST   # candidates -> ids_pool.txt
sbatch --wait -J af-real -a 0-9 scripts/eval/action_follow_eval.sbatch moved real rec real "${AF[@]}" \
  --ids "$A/ids_pool.txt" --shard TASK 10
"$PY" -m simpwm.eval.action_follow select "${AF[@]}" --test-ids $TEST   # -> ids100.txt, selection.json

# 2. counterfactual robot inputs (noclose, offset) and their proxies
"$PY" -m simpwm.droid.run_counterfactual prep "$A/ids100.txt" --ref-run "$REF" --out "$A"
for c in noclose offset; do
  sbatch --wait -J af-proxy -a 0-9 scripts/eval/action_follow_proxy.sbatch $c "$A/ids100.txt" 10 "$A/px_$c" "$A/actions" &
done
wait_all

# 3. proxy readout; SimPWM videos: recorded actions = the Table 3 test videos, the conditions rendered from their proxies
"$PY" -m simpwm.eval.action_follow readout rec "$REF" "${AF[@]}"
for c in noclose offset; do
  "$PY" -m simpwm.eval.action_follow readout $c "$A/px_$c" "${AF[@]}"
  "$PY" -m simpwm.eval.action_follow build-ds $c "$A/px_$c" "${AF[@]}"
done
for m in rgb robot depth; do
  mkdir -p "$A/videos/simpwm_${m}_rec"
  for i in $(cat "$A/ids100.txt"); do ln -sfn "$T3/videos/$m/$i.mp4" "$A/videos/simpwm_${m}_rec/$i.mp4"; done
  for c in noclose offset; do
    sbatch --wait -J af-gen -a 0-4 scripts/droid/generate.sbatch "$TRAIN/$m.yaml" \
      "$TRAIN/$m/checkpoints/lora_weights_step_$STEP.safetensors" "$A/ds_$c" "$A/gen/${m}_$c" 5 &
  done
done
wait_all
for m in rgb robot depth; do
  for c in noclose offset; do "$PY" -m simpwm.eval.action_follow export "$A/gen/${m}_$c" "$A/videos/simpwm_${m}_$c" "${AF[@]}"; done
done

# 4. both judges on every row x condition (and on the recordings), then agreement, review set, summary, figures
specs=(real:rec:real:1)
for r in ctrl_world:3 dreamdojo:1 simpwm_rgb:1 simpwm_robot:1 simpwm_depth:1; do
  for c in rec noclose offset; do specs+=("${r%%:*}:$c:$A/videos/${r%%:*}_$c:${r##*:}"); done
done
sbatch --wait -J af-objects scripts/eval/action_follow_eval.sbatch objects "${AF[@]}"
sbatch --wait -J af-moved -a 0-19 scripts/eval/action_follow_moved.sbatch "$A" "$REF" 20 "${specs[@]}" &
sbatch --wait -J af-vlm -a 0-9 scripts/eval/action_follow_vlm.sbatch "$A" 10 "${specs[@]}" &
wait_all
for s in follow review summary figures; do "$PY" -m simpwm.eval.action_follow $s "${AF[@]}"; done
