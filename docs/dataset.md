# Dataset

The DROID data of the experiments: the raw episodes, first-frame stereo depth, cameras, proxies, the test set and the
episode selection. Training is in [train.md](train.md), generation in [inference.md](inference.md). Commands run from the repository root with the main environment (see the README);
`scripts/env.sh` sets `PYTHONPATH` and the defaults below. Slurm scripts are submitted from the repository root and
log to `logs/`.

| what | where (default) | override |
|---|---|---|
| raw DROID episodes | `data/droid_raw/` | `DROID_RAW` |
| language annotations | `data/droid_language_annotations.json` | `SIMPWM_DATA` |
| downloaded stereo recordings | `data/droid_raw_stereo/` | `SIMPWM_DATA` |
| outputs | `runs/` (`runs/stereo`, `runs/droid`) | `SIMPWM_RUNS` |
| LTX-2 trainer, FoundationStereo | `deps/LTX-2`, `deps/FoundationStereo` | |
| LTX-2.3 and Gemma weights | `models/ltx-2.3/ltx-2.3-22b-dev.safetensors`, `models/gemma-3-12b-it-qat-q4_0-unquantized` | |

## 1. Data

One directory per episode of the public raw release (`gs://gresearch/robotics/droid_raw/1.0.1`, plain HTTPS), named by
the episode uuid:

```
$DROID_RAW/<lab>+<user>+<date>-<time>/   ext1.mp4  ext2.mp4  trajectory.h5  metadata.json
```

`ext1.mp4` / `ext2.mp4` are the release's `recordings/MP4/<serial>.mp4` (1280x720) of the metadata's `ext1_cam_serial` /
`ext2_cam_serial`, `metadata.json` its `metadata_*.json`. The instruction is `language_instruction1` of
`droid_language_annotations.json` (Hugging Face dataset `KarlP/droid`), else the metadata's `current_task`.

`simpwm/droid/data.py` resizes the frames to 320x180 and takes joints and the observed gripper position from
`trajectory.h5` (frame i = step i; the recordings have one frame fewer than the trajectory). An **episode index** is a
line number of `resources/droid/episodes.txt`, the 42,927 usable episodes of the release, so every shipped index list
is valid whatever subset of the release is downloaded. If the RLDS release with its
manifest is found under `DROID_ROOT` (`1.0.1/`, `raw_lowdim_1.0.1/manifest.jsonl`), it is used instead; its indices
differ, and the video datasets need the raw source.

Shipped in `resources/droid/`:

| file | contents |
|---|---|
| `splits/{train,val,test}.txt` | the paper's episodes: 3000 train, 8 validation, 500 test (setups without training episodes) |
| `episodes.txt` | the 42,927 usable episodes of the raw release; line i = episode index i |
| `cameras/<idx>.json` | fitted per-episode cameras of 6,394 episodes, including every train / val / test episode (step 3) |
| `setup_cameras/<key>.json` | per-setup camera calibrations (674 setups) |
| `droid500.json`, `video_train2k.json` | setup-level sets read by the episode selection (step 6) |

## 2. First-frame stereo depth

The ext1 stereo pair of the first frame gives the metric depth of the scene. Three environments are involved:

| stage | environment | setup |
|---|---|---|
| download, `rlds` (first frames) | main | none |
| `svo` (rectified pairs from the ZED recordings) | ZED SDK, `ZED_HOME` | extract `ZED_SDK_Ubuntu24_cuda12.8_tensorrt10.9_v5.0.7.zstd.run --noexec --target $ZED_HOME/ex` (no root needed); `libturbojpeg.so.0` into `$ZED_HOME/deps/lib`; `uv venv -p 3.11 $ZED_HOME/venv` with the `pyzed-5.0` cp311 wheel (download.stereolabs.com/zedsdk/5.0/whl/linux_x86_64), `"numpy<2" opencv-python-headless imageio`. Factory calibrations are fetched into `$ZED_HOME/settings` on first use (needs internet) |
| `fs` (FoundationStereo depth) | FoundationStereo venv, `FS_PY`, `FS_SHIM` | `git clone https://github.com/NVlabs/FoundationStereo deps/FoundationStereo`; `uv venv -p 3.11` with `torch==2.4.1 torchvision==0.19.1` (cu121), `timm einops omegaconf scipy opencv-python-headless imageio trimesh pandas joblib huggingface-hub scikit-image`; `FS_PY` = its python; `FS_SHIM` = a directory holding an empty `open3d/__init__.py` (imported by FoundationStereo's utilities, unused for depth). The ViT-L checkpoint (`shriarul5273/FoundationStereo_models`) and DINOv2 (`torch.hub`) are downloaded on first use; the launchers run offline, so run the first `fs` task with `HF_HUB_OFFLINE=0` |

```bash
export ZED_HOME=... FS_PY=... FS_SHIM=...
scripts/droid/stereo_depth.sh IDS_TXT       # -> runs/stereo/depth/<idx>_ext1.npy
```

`stereo_depth.sh` downloads the ext1 `.svo` recordings (`python -m simpwm.droid.stereo.download --episodes IDS_TXT --svo
ext1`, ~8 MB per episode) and runs the stages as Slurm arrays (`scripts/droid/stereo.sbatch {rlds,svo,fs} 16`). The `svo`
stage takes `.svo` frame 0 when the recording and the trajectory have the same length, else the frame that best matches
the loader's first frame; upside-down cameras are detected and their views swapped.

## 3. Cameras

Camera (world = robot base frame) of an episode, in `simpwm/droid/run.py`'s order:

1. **Per-episode camera** `--camera-dir DIR` (`DIR/<idx>.json`). You do not need to fit cameras: the fitted cameras
   of 6,394 episodes, including every episode of the paper's splits, are shipped in `resources/droid/cameras`, and the
   training and test proxies use them. They were fitted on each episode's own recording with
   `simpwm.droid.episode_calib`: SAM 3 robot masks over the clip and the rest of the episode, robot-silhouette fits
   from the setup camera and from the logged extrinsic, kept only if the arm IoU over the clip improves.
2. `--logged-camera` (ablation): the camera DROID logged.
3. **Setup camera** (`simpwm/droid/setup_calib.py`): calibrated once per camera setup (lab, ext1 serial, logged ext1
   extrinsic) from robot silhouettes in up to 6 *other* episodes of the setup, cached in `resources/droid/setup_cameras`;
   a missing setup is calibrated on first use. The setup members come from `data/droid_setup_index.json`, built on
   first use, or from `--setups JSON`.

## 4. Proxies

```bash
python -m simpwm.droid.run OUT_DIR IDX [IDX ...] [--camera-dir DIR] [--stereo-dir runs/stereo/depth] \
    [--ref-size 1280x704] [--clip-frames 97] [--clip-only]
```

Per episode `OUT_DIR/droid_XXXXX/`: `check.mp4` (real | overlay / depth | categories), `conditions/` (24 fps proxy
frames; with `--ref-size` also the first `--clip-frames` steps at 15 fps, rendered at the recording's resolution:
`ref_1280x704.mp4` control video (inverse depth, robot, held object), `ref_rgb_1280x704.mp4` RGB proxy,
`ref_depth_1280x704.mp4` depth proxy), `metrics.json` / `metrics_sam3.json` (objects and robot tracked in the real
video vs. the proxy), `proxy_cat.npz`, `info.json` (written last; finished episodes are skipped). Without
`--stereo-dir` the first-frame depth is monocular; with it, episodes without stereo depth are skipped.
`--clip-only` simulates only the clip and skips the evaluation (training proxies, ~3 min per episode).

`scripts/droid/proxy.sbatch IDS_TXT N_TASKS OUT_DIR [run options]` runs shard `SLURM_ARRAY_TASK_ID` with three processes
on one GPU, with stereo depth (`STEREO_DIR`, empty = monocular) and `--ref-size 1280x704`.

**High-resolution proxies.** `--hires` selects the `droid_hires` preset ([configuration.md](configuration.md)): the
simulation runs at 640x360, and SAM 3, the object colours and the background depth use the 720p first frame. It needs
`--ref-size` and the raw DROID release. SimPWM-H3 and the examples of the project page use these proxies, with the
cameras of `resources/droid/cameras_hires`:

```bash
python -m simpwm.droid.run OUT_DIR IDX [IDX ...] --hires --camera-dir resources/droid/cameras_hires \
    --stereo-dir runs/stereo/depth --ref-size 1280x704 --clip-frames 97 --clip-only
```

Deformable objects: `python -m simpwm.perception.vlm_objects IDX ...` names what the gripper closes on (Qwen3-VL) into
`runs/vlm_objects`; the `soft_objects` rule applies to the episodes that have such a file (`--vlm-dir`) and is
inactive without it.

Simulator ablations (paper Table 2) are switches of `simpwm/config.py`, set from the environment, e.g. `SIMPWM_SNAP=0`,
`SIMPWM_SAM3_PROPOSALS=0`; `--logged-camera` and the monocular depth are run options.

**Proxy videos.** The videos written with `--ref-size WxH` (and by `simpwm/gr1/run.py`) use the
functions of `simpwm/video/dataset.py`; fixed ranges, so a value means the same in every episode:

| video | encoding |
|---|---|
| control `ref_WxH.mp4` | R = inverse depth over 0.2-3 m (near = bright, none / beyond 3 m = 0), G = robot (Lambert-shaded mesh luminance, floor 0.25), B = held objects (`control_frame`) |
| RGB proxy `ref_rgb_WxH.mp4` | the first frame of the raw 720p recording as the static scene, the robot mesh and the held objects (first-frame point colours) drawn over it; their uncovered first-frame footprint is magenta, "unknown, to be completed" (`world_rgb_frame`) |
| depth proxy `ref_depth_WxH.mp4` | the R channel of the control video as a grey image (`depth_frame`); `decode_depth` inverts it |
| robot only `refs_robot/` | the G channel of the control video (`python -m simpwm.video.robot_refs`) |

## 5. Test set

```bash
scripts/droid/build_test_set.sh                 # -> runs/droid/datasets/test
```

Stereo depth of `splits/test.txt`, proxies with `--camera-dir resources/droid/cameras --clip-frames 241 --clip-only`
into `runs/droid/proxies`, the link-mode test dataset ([train.md](train.md#1-paired-datasets)) and its robot-only
references. [inference.md](inference.md) renders it.

## 6. Episode selection

The splits were chosen with `python -m simpwm.droid.select` (indices of the full raw release):

```bash
S=runs/droid/select
python -m simpwm.droid.select candidates $S --per-setup 12   # >= 241 steps, setups outside DROID-500 dev/test
cat $S/ids_val.txt $S/ids_train.txt > $S/ids_cand.txt
scripts/droid/stereo_depth.sh $S/ids_cand.txt
sbatch --wait -a 0-47 scripts/droid/calib_episodes.sbatch $S/ids_cand.txt 48 $S/cameras --clip-frames 241
python -m simpwm.droid.select pick $S --camera-dir $S/cameras --stereo-dir runs/stereo/depth   # arm IoU >= 0.6: 3000 + 8
python -m simpwm.droid.select test $S                         # setups without training episodes, grasp by step 72
# stereo depth and camera fits of $S/ids_test_cand.txt as above, into $S/test_cameras
python -m simpwm.droid.select pick-test $S --camera-dir $S/test_cameras --stereo-dir runs/stereo/depth   # 500
```

`pick` takes the fitted candidates in their round-robin order; the paper's pick had camera fits for 5,461 of the 7,179
candidates, so fitting all of them can select different episodes. `splits/` holds the paper's episodes.

## 7. Counterfactual actions

`python -m simpwm.droid.run_counterfactual prep IDS_TXT --ref-run runs/droid/proxies` computes the perturbed robot
inputs of the action-following experiment (`noclose`: the gripper never closes; `offset`: the end effector shifted 5 cm
towards the image right, joints through IK) into `runs/action_follow/actions`;
`python -m simpwm.droid.run_counterfactual run {noclose,offset} OUT_DIR IDX ... [run options]` renders their proxies with
`simpwm.droid.run`'s options (those of the test set: `--camera-dir resources/droid/cameras --stereo-dir
runs/stereo/depth --ref-size 1280x704 --clip-frames 241 --clip-only`).
