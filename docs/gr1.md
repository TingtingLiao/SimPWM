# GR1

Reproduction of the GR1 humanoid experiment (supplementary GR1 table): the sim-to-real step only (the rasterized robot
over the warped first frame, no simulated objects), the LTX-2 IC-LoRA trained on 3,000 clips at 800x480 and 10 fps,
and the comparison with DreamDojo on its 100 test windows. Commands run from the repository root with the main
environment (see the README) plus `pyarrow` (`pip install -e .[gr1]`); `scripts/env.sh` sets `PYTHONPATH` and the
defaults below. Slurm scripts are submitted from the repository root.

| what | where (default) | override |
|---|---|---|
| GR00T-Teleop-GR1, subset `GR1_robot` (LeRobot v2) | `data/gr1/GR1_robot/` | `GR1_ROOT` |
| GR1T1 URDF with Fourier hands | `assets/grx_models/` | `SIMPWM_ASSETS` |
| outputs | `runs/gr1/` | `SIMPWM_RUNS` / script arguments |
| LTX-2 trainer | `deps/LTX-2` | `SIMPWM_DEPS` |
| LTX-2.3 and Gemma weights | `models/ltx-2.3/ltx-2.3-22b-dev.safetensors`, `models/gemma-3-12b-it-qat-q4_0-unquantized` | |
| DreamDojo (baseline) | `deps/DreamDojo` | `DREAMDOJO`, `DD_PY`, `DD_CKPT` |

## 1. Data

```bash
hf download nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1 --repo-type dataset --include "GR1_robot/*" --local-dir data/gr1
git clone https://github.com/FFTAI/Wiki-GRx-Models assets/grx_models    # GRX/GR1/gr1t1/urdf/gr1t1_fourier_hand_6dof.urdf
```

`simpwm/gr1/data.py` reads `meta/episodes.jsonl`, the 44-d `observation.state` of `data/chunk-*/episode_*.parquet` and the
ego-view video (`videos/chunk-*/observation.images.ego_view_freq20`, 1280x800, 20 fps). Clips are sampled at stride 2
(10 fps, DreamDojo's GR1 sampling). The caption is "A head-mounted camera films a Fourier GR1 humanoid robot with two
dexterous hands working at a table. The robot performs the task: <task>. Realistic lighting, natural colors."

Shipped in `resources/`:

| file | contents |
|---|---|
| `gr1/dreamdojo_test_split.json` | DreamDojo's GR1_robot test split (`data_split=test`): its episodes (21705-22208) and 100 windows |
| `gr1/test_windows.json` | the 100 test windows `{k, ep, base, start}`: 49 frames, raw frames `base + 2 + 2j` |
| `gr1/sessions.json` | per recording session (GR1 unit + start time), up to 3 non-test episodes for the camera refit |
| `gr1/session_cameras/<session>.json` | the per-session head cameras of step 3 (the ones used in the paper) |
| `embodiments/gr1_head_camera.json` | the dataset-level head camera with per-unit and per-session refinements (step 3's start) |

## 2. Item selection

```bash
python -m simpwm.gr1.select                     # -> runs/gr1/{candidates,sessions,test_windows}.json
```

`candidates.json` lists 4,800 training clips `{ep, start, session}`: non-test episodes with at least 195 frames,
round-robin over sessions, a random clip start. The selection is seeded; `sessions.json` and `test_windows.json` equal
the shipped files. No test episode is used for calibration or training.

## 3. Head camera and hand scale

The head camera moves with the waist and neck: `T_wc(t) = FK(head_pitch_link)(t) @ T_hc`, with a fixed head->camera
offset, focal length and radial distortion per recording session (`simpwm/gr1/calib.py`). Each session's camera is refined
from the stored calibration on its `sessions.json` episodes (12 frames each, SAM 3 robot masks, GPU rasterizer): the 9
camera parameters by Nelder-Mead on the pooled robot IoU, then the state/video time offset `dt` (-6..6 raw frames),
kept only if the IoU improves.

```bash
sbatch -a 0-15 scripts/gr1/calib_sessions.sbatch 16           # -> runs/gr1/session_cameras/<session>.json
```

The proxies read `resources/gr1/session_cameras` unless `--cameras` points elsewhere. The Fourier-hand joint mapping
under-closes the right hand; the per-hand closure scale is fitted per episode, jointly with the camera, by

```bash
python -m simpwm.gr1.hand_scale runs/gr1/hand_scale --episodes 16    # GPU; prints the fitted (left, right) scales
```

The paper uses `--hand-scale 0.89:1.28` (set by `scripts/gr1/proxy.sbatch`).

## 4. Proxies

```bash
python -m simpwm.gr1.run {train,eval} OUT [--items JSON] [--shard K --num-shards N] [--cameras DIR] \
    [--hand-scale L:R] [--min-iou 0.6] [--no-episode-refit]
sbatch -a 0-31 scripts/gr1/proxy.sbatch train 32               # runs/gr1/candidates.json -> runs/gr1/proxies
sbatch -a 0-3 scripts/gr1/proxy.sbatch eval 4                  # test windows -> runs/gr1/proxies_eval
```

Per item (`<ep>` for training clips of 97 frames, `w<k>` for test windows of 49 frames): `video.mp4` (real clip,
1280x768, 10 fps), `first.png`, `ref.mp4` (RGB proxy, 800x480: the robot rendered from the per-frame head camera over
the first frame, which is lifted to 3D with Depth Anything V2 scaled by the FK robot and re-splatted into the moving
camera; magenta where nothing is known), `prompt.txt`, `info.json` (arm IoU, camera) and, for test windows, `robot.npz`
(robot masks for the evaluation). Training clips refit the camera on their own frames and are skipped below arm IoU
0.6; 3,004 passing clips are needed (3,000 training + 4 validation).

## 5. Training data and training

```bash
scripts/gr1/build_training_data.sh                             # runs/gr1/{ds_train,ds_val,ds_eval}, runs/gr1/train/cfg.yaml
sbatch -a 0-15 scripts/gr1/latents.sbatch runs/gr1/ds_train 16 runs/gr1/latents
sbatch -N 4 scripts/gr1/train.sbatch runs/gr1/train/cfg.yaml   # 4 nodes x 4 GPUs
```

1. link-mode datasets (`python -m simpwm.video.dataset_gr1`): the first 3,000 passing candidates (train), the next 4
   (validation samples during training), the 100 test windows (eval);
2. latents of the training set at 800x480x97 (LTX-2's `process_dataset.py`: video, caption, RGB proxy at downscale
   factor 1, 10 fps);
3. IC-LoRA config (`python -m simpwm.video.make_config ... --preset gr1`: RGB proxy reference, 2,000 steps, lr 2e-4) and
   DDP training (LTX-2's `scripts/train.py` with the trainer patch, `SIMPWM_REF_POS_FIX=1`).

## 6. Generation

```bash
sbatch -a 0-7 scripts/gr1/generate.sbatch runs/gr1/train/cfg.yaml \
    runs/gr1/train/checkpoints/lora_weights_step_02000.safetensors runs/gr1/ds_eval runs/gr1/gen 8
```

Each shard writes its config (`python -m simpwm.video.eval_config --clip-frames 49`: 49 frames at 800x480, 10 fps, RGB proxy reference, real
first frame) and runs the trainer's validation pipeline: `runs/gr1/gen/shard_T/samples/step_000000_<k>.mp4` is the k-th
window of `shard_T/order.json`, as a [reference | generation] panel.

## 7. DreamDojo baseline

Clone `https://github.com/NVIDIA/DreamDojo` into `deps/DreamDojo` and install its environment (its README); download its
GR1 post-trained 2B checkpoint (`nvidia/DreamDojo` on Hugging Face, `2B_GR1_post-train`, iteration 50000).

```bash
DD_PY=/path/to/dreamdojo/python DD_CKPT=deps/DreamDojo/checkpoints/2B_GR1_post-train \
    sbatch -a 0-7 scripts/gr1/dreamdojo.sbatch 8                # -> runs/gr1/dreamdojo/shard_T/iter_000050000/<k>_{gt,pred}.mp4
```

This is DreamDojo's own `examples/action_conditioned.py`, unmodified, with its evaluation settings (`--num-frames 49
--num-samples 100 --data-split test --deterministic-uniform-sampling`) on the same `GR1_robot` copy;
`scripts/gr1/run_dreamdojo.py` only applies three environment fixes (unserved Hugging Face revision of the Wan2.1 VAE,
unused base-checkpoint downloads, CPU affinity under Slurm).

## 8. Evaluation

```bash
pip install piq==0.8.0                                         # DreamDojo's metric package
scripts/gr1/evaluate.sh                                        # -> runs/gr1/bench.json, runs/gr1/grids/
```

`python -m simpwm.eval.gr1_bench score` scores both models against the same ground truth, DreamDojo's `<k>_gt.mp4`, with
its evaluation transform (4:3 centre crop of the 1280x800 frame, 640x480); our 800x480 generation is resampled to the
same field of view. The table reads `summary` of `bench.json`: PSNR / SSIM / LPIPS = `{psnr,ssim,lpips}.{simpwm,dreamdojo}
.mean` (piq, 49 frames per window), FVD = `fvd.{simpwm,dreamdojo}` (I3D, 16 frames; only indicative on 100 clips), robot
IoU = `robot_iou.{simpwm,dreamdojo}` (SAM 3 robot masks of the generated vs. the real frames, frames 0, 8, ..., 48).
`checks` verifies the ground truth against the transform of the raw video and the frame-0 alignment. `gr1_bench grids`
draws real / SimPWM / DreamDojo / proxy frames of the windows with the lowest, median and highest SimPWM PSNR.
