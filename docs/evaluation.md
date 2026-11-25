# Evaluation

The paper's DROID evaluations (500 test episodes from setups without training episodes,
`resources/droid/splits/test.txt`), the modules that compute them and how to run them. GR1 is evaluated
as described in [gr1.md](gr1.md). Commands run from the repository root
(`source scripts/env.sh`); Slurm launchers in `scripts/eval/` take the arguments of their module, with the word `TASK`
replaced by the array task id (e.g. `--shard TASK 20`).

| paper | what | modules | driver | output |
|---|---|---|---|---|
| Table 1 | world-model benchmark: our model vs. the baselines | `wm_bench` (`--protocol table1`), `grasp_vlm` | commands below | `bench_summary.json`, `vlm/summary.json` |
| Table 2 | real-to-sim proxy ablation (rule-based rows, PointWorld) | `real_dyn`, `proxy_metrics`, `arm_iou`, `grasp_iou`, `grasp_vlm`, `grasp_vlm_ablation` | `scripts/eval/proxy_ablation.sh` | `runs/table2/summary.json` |
| Table 3 | sim-to-real ablation of the video model; VLM grasp judge | `wm_bench` (`--protocol table3`), `grasp_vlm`, `grasp_vlm_follow`, `grasp_vlm_notext` | `scripts/eval/video_ablation.sh` | `runs/table3/bench_summary.json`, `runs/table3/vlm/summary.json` |
| Table 4 | action following under counterfactual actions | `action_follow` (+ `simpwm.droid.run_counterfactual`) | `scripts/eval/action_follow.sh` | `runs/action_follow/summary.json` |

All modules are `simpwm.eval.<module>`; `python -m simpwm.eval.<module> --help` lists their options.

## Inputs

| input | default | produced by |
|---|---|---|
| test proxies of the full system (reference run) | `runs/droid/proxies` | `scripts/droid/build_test_set.sh` ([dataset.md](dataset.md) 5) |
| test dataset (real videos, first frames, captions, proxies) | `runs/droid/datasets/test` | same |
| first-frame stereo depth | `runs/stereo/depth/<idx>_ext1.npy` | [dataset.md](dataset.md) 2 |
| later-step stereo depth (steps 24, 48, 72, 96) | `runs/stereo_seq/depth/<idx>_ext1_t<t>.npy` | `simpwm.droid.stereo.extract svoseq --out runs/stereo_seq` on the paired test recordings, then its `fs` stage with `--out runs/stereo_seq` |
| trained models | `runs/droid/train/<model>.yaml`, `<model>/checkpoints/` | [train.md](train.md) 2 |
| VLM judge | `Qwen/Qwen3-VL-8B-Instruct` (`--vlm`) | Hugging Face cache (jobs run offline) |
| external benchmark code | `deps/benchmarks/` (`--bench-root`) | `WorldArena/`, `physics-IQ-benchmark/` (their repositories), `fvd/i3d_torchscript.pt` (StyleGAN-V's I3D), `pylib/` (`pip install --target deps/benchmarks/pylib fastdtw lpips`) |
| baseline videos | `<dir>/<idx>.mp4` per model | the baselines' own code on the test episodes (Ctrl-World at 5 Hz: `--stride 3`) |

## Per-run proxy metrics

`simpwm/droid/run.py` scores every proxy run against the real video with `tracks.py` (SAM 2 tracks: robot IoU, object IoU
and centroid error next to an "object never moves" baseline) and `tracks_sam3.py` (the same with SAM 3's tracker);
`arm_iou.py` (SAM 3 robot instances touching the render, kept whole; pixel-weighted IoU over the clip) is the
camera-alignment metric of the training-set selection and of Table 2:

```bash
sbatch -a 0-31 scripts/eval/arm_iou.sbatch RUN --ids IDS_TXT --shard TASK 32 --clip-frames 241
sbatch -a 0-31 scripts/eval/grasp_iou.sbatch RUN --ids IDS_TXT --shard TASK 32 --clip-frames 241
```

`grasp_iou.py`: which object the real gripper grasps (a closure that stalls short of fully closed; SAM 2 segments the
thing between the fingers and tracks it back to step 0; strict mode by default) against the objects the proxy holds;
grasp success (same object or neither grasps), object IoU over the clip (SAM 3 track of the target), end error.

## Table 1: world-model benchmark

`wm_bench.py` scores directories of generated test videos against the real ones under one protocol for every row:
`table1` reduces every video to 320x192 at recorded steps 0, 3, ..., 96 (5 Hz, the coarsest compared model's output).
Metrics: PSNR, LPIPS, FVD (I3D, jackknife interval), robot IoU (SAM 3 instances picked by the reference run's robot
render), trajectory accuracy (WorldArena), depth error (Depth Anything V2 median-scaled to stereo), Physics-IQ, Interact.
(WorldArena's VLM judge) and the deformation IoU (episodes whose instruction names a deformable object). Means with
bootstrap 95% intervals over the episodes every row has.

```bash
T=resources/droid/splits/test.txt; DS=runs/droid/datasets/test; B=runs/table1
COMMON="--ids $T --test-set $DS --protocol table1"
sbatch --wait -a 0-24 scripts/eval/wm_bench.sbatch real $B $COMMON --ref-run runs/droid/proxies --shard TASK 25
for row in simpwm:VIDEOS:1 ctrl_world:VIDEOS:3 dreamdojo:VIDEOS:1 cosmos_predict25:VIDEOS:1; do   # NAME:DIR:STRIDE
  IFS=: read -r name dir stride <<< "$row"
  sbatch -a 0-24 scripts/eval/wm_bench.sbatch gen $B $name $dir $COMMON --ref-run runs/droid/proxies --stride $stride --shard TASK 25
  sbatch -a 0-4 scripts/eval/wm_bench.sbatch vlm $B $name $dir $COMMON --stride $stride --shard TASK 5
done
python -m simpwm.eval.wm_bench summary $B $B/bench_summary.json simpwm ctrl_world dreamdojo cosmos_predict25 --protocol table1
```

The VLM grasp judge of Table 3 (below) runs on the baselines' videos the same way (`grasp_vlm answer ... --stride 3` for
Ctrl-World).

## Table 2: proxy ablation

Rows: the full system (the test proxies) and the simulator ablated one piece at a time, all on the 97-step protocol
(the first 97 recorded steps), plus PointWorld's point-flow proxies:

| row | proxy run (`simpwm.droid.run` on the test set, as the full system unless noted) |
|---|---|
| full | `runs/droid/proxies` |
| cam | `--logged-camera`, no `--camera-dir` (DROID's logged camera) |
| mono | `STEREO_DIR=` (monocular first-frame depth) |
| snap | `SIMPWM_SNAP=0 SIMPWM_PIN=0` (no action-conditioned depth snap) |
| sam3 | `SIMPWM_SAM3_PROPOSALS=0` (no SAM 3 object proposals) |
| pw | PointWorld rollouts (`<idx>.npz`: first-frame pixels, per-step point positions, camera) and their RGB proxy videos, rendered with the full system's robot |

```bash
ROWS="full=runs/droid/proxies cam=RUN mono=RUN snap=RUN sam3=RUN" PW_ROLLOUTS=DIR PW_VIDEOS=DIR \
    scripts/eval/proxy_ablation.sh runs/table2
```

1. `real_dyn.py`: real-video masks shared by every row (SAM 3 robot instances; the strict grasp targets from the full
   system's camera, tracked with SAM 3).
2. `proxy_metrics.py run`: arm IoU, RGB L1 / L2 over the moving region (real robot and targets united with the proxy's
   robot and held objects, magenta excluded), change-masked depth L2 against later-step stereo depth;
   `grasp_iou.py --lenient --clip-frames 97 --gt-run FULL`: automatic grasp success and object IoU against the full
   system's targets; `proxy_metrics.py summary` aggregates both per row (`auto.json`).
3. `grasp_vlm.py` on each row's RGB proxy video (Grasp / Halluc. columns), matched against Table 3's real-video answers.
4. `grasp_vlm_ablation.py`: the image object IoU of every row, PointWorld's RGB / depth metrics, and all rows scored on
   common pixels (`aligned`); `summary` writes `summary.json` with every number of the table.

## Table 3: sim-to-real ablation and the grasp judge

Video models conditioned on different proxies (no proxy, robot only, PointWorld, RGB, depth, RGB + depth; trained as
in [train.md](train.md) 2) on the test set:

```bash
scripts/eval/video_ablation.sh runs/table3 noproxy robot pw rgb depth rgb_depth
```

1. test videos of every model (step-2000 LoRA, `scripts/droid/generate.sbatch`), cut to `videos/<model>/<idx>.mp4`;
2. `wm_bench.py --protocol table3`: the Table 1 metrics at the models' native output (1280x704, all 97 steps);
3. `grasp_vlm.py`, the VLM grasp judge: Qwen3-VL-8B-Instruct, greedy, 8 frames at recorded steps 0, 14, ..., 96
   resized to 1280 wide, "Does the robot pick up and lift an object off its support in these frames? If yes, name the
   object; if no, answer none."; a deterministic parse of the answer; a generation agrees with the real video when both
   say none, or name the same object (identical names, or the text-only VLM judges the two names the same object given
   the task). Grasp acc. = agreement, Halluc. = an object named where the real video has none, Miss = the converse.

Analyses of the judge's answers:

```bash
# which grasp the generations follow when their proxy and the real grasp disagree
sbatch scripts/eval/grasp_vlm_follow.sbatch runs/table3/follow.json runs/table3/vlm --test-set runs/droid/datasets/test \
    --proxy rgb=runs/table2/vlm/full --proxy depth=runs/table2/vlm/full --proxy rgb_depth=runs/table2/vlm/full \
    --proxy pw=runs/table2/vlm/pw
# the same models without the task sentence in the caption: generate on a copy of the test set whose captions end
# after the scene sentence, judge with grasp_vlm (answer + match into runs/table3_notext/vlm), then
sbatch scripts/eval/grasp_vlm_notext.sbatch runs/table3_notext/summary.json --ids IDS_TXT --with-text runs/table3/vlm \
    --no-text runs/table3_notext/vlm --proxy runs/table2/vlm/full --test-set runs/droid/datasets/test
```

## Table 4: action following

100 test episodes in which the real robot picks up an object, under three robot inputs: the recorded actions (`rec`),
the gripper never closing (`noclose`) and the end effector shifted 5 cm towards the image right (`offset`). Rows:
Ctrl-World, DreamDojo (their videos rendered from `ROOT/actions` by their own code), SimPWM with the robot-only, depth and
RGB proxy, and the SimPWM proxy readout (the simulator's own held state).

```bash
scripts/eval/action_follow.sh runs/action_follow
```

1. samples: test episodes whose gripper stalls on something in the first 97 steps with a strict grasp target that the
   real robot visibly moves (`stalled`, `targets`, `select`; 100 drawn with seed 0);
2. counterfactual robot inputs and their proxies (`simpwm.droid.run_counterfactual prep / run`);
3. the SimPWM videos of each condition (`build-ds` + `scripts/droid/generate.sbatch`; `rec` = the Table 3 test videos);
4. judges on 33 frames per video at 640x352: the primary VLM judge ("Does the robot gripper pick up / hold and lift the
   {object}?" on 8 frames from the close time to step 96) and the supplementary "moved" rule (the target's SAM 2 track
   moves > 30 px and lies within 40 px of the gripper); Grasp = fraction yes under `rec`, Halluc. = under `noclose` /
   `offset`. `follow` measures the agreement of the proxy readout with the generated videos; `summary` writes the table.

## Module reference

| module | role |
|---|---|
| `common.py` | DROID episode access, proxy-run cameras, frame readers, id lists, bootstrap intervals |
| `vlm.py` | the Qwen3-VL judge (loading, greedy chat) |
| `tracks.py`, `tracks_sam3.py` | SAM 2 / SAM 3 video tracking; per-run proxy metrics |
| `arm_iou.py` | robot alignment of a proxy run |
| `grasp_gt.py`, `grasp_iou.py` | real grasps and their targets; grasp success and object IoU of a proxy run |
| `real_dyn.py`, `proxy_metrics.py` | Table 2's shared real masks and per-run metrics |
| `grasp_vlm_ablation.py` | Table 2's PointWorld row, image object IoU, aligned metrics and summary |
| `wm_bench.py` | Tables 1 and 3: world-model metrics of generated videos |
| `grasp_vlm.py`, `grasp_vlm_follow.py`, `grasp_vlm_notext.py` | the VLM grasp judge and its analyses |
| `action_follow.py` | Table 4 |
