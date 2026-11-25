# Training

Training a SimPWM video model takes three steps: simulate the training episodes into proxies, pair each proxy with its
real video, and train a LoRA on the video model.

| Model | Base model | Proxy | Training set | Hardware |
|---|---|---|---|---|
| SimPWM (paper, Tables 1-4) | LTX-2.3 (22B), IC-LoRA | `droid` preset, 320x180 simulation | 3000 episodes, `resources/droid/splits/train.txt` | 8 A100 80 GB, ~13 s / step, 2000 steps |
| SimPWM-H3 | MiniMax-H3 Ref2VA (33B), LoRA | `droid_hires` preset, 640x360 simulation | 845 episodes, `resources/droid/splits/train_h3.txt` | 32 A100 80 GB, ~140 s / step, 250 steps |

Both render 97-step clips (6.5 s at 15 Hz) at 1280x704 from the RGB proxy, the real first frame and the task text.

Before you start: the main environment and the video model ([installation.md](installation.md), sections 1-4), the raw
DROID release and the first-frame stereo depth ([dataset.md](dataset.md), sections 1-2). The cameras are shipped, so
nothing has to be fitted. The launchers are Slurm scripts, submitted from the repository root.

## 1. Paired datasets

All follow the ltx-trainer convention: `OUT/dataset.jsonl` rows `{caption, video, reference_video, idx}` with paths
relative to `OUT`, `refs/` (proxy), `videos/` (real video), `first/` (real first frame at the generation size).

| dataset | module | layout |
|---|---|---|
| DROID | `python -m simpwm.video.dataset RUN OUT @IDS --link --size 1280x704 --clip-frames 241` | `--link`: symlinks to the raw ext1 MP4 (1280x720, 15 fps) and the RGB / depth proxies (`refs/`, `refs_depth/`, `dataset_depth.jsonl`); without it the control video and the real video (from the raw MP4) are re-encoded (`--raw`: at the recording's resolution and rate) |
| DROID, robot only | `python -m simpwm.video.robot_refs RUN DS --clip-frames 97` | `DS/refs_robot/`, `DS/dataset_robot.jsonl` |
| GR1 | `python -m simpwm.video.dataset_gr1 {train,val,eval} PX OUT --items ITEMS` | symlinks to the item's `video.mp4` (1280x768, 10 fps), `ref.mp4` (800x480), `first.png`; train / val keep items with arm IoU > 0.6 |

Captions: "A static camera films a Franka robot arm with a two-finger gripper working on a table in a lab. The robot arm
performs the task: <instruction>. Realistic lighting, natural colors." (DROID).

To train on your own recordings, write the same layout; the proxy comes from the simulator, and `simpwm/demo.py` shows
the inputs it needs (first frame, camera, actions).

## 2. LTX-2.3 IC-LoRA

### Training data

```bash
scripts/droid/build_training_data.sh            # -> runs/droid
```

On the train + val split, waiting for each Slurm array:

1. stereo depth ([dataset.md](dataset.md#2-first-frame-stereo-depth));
2. proxies of the first 241 steps with the shipped cameras (`proxy.sbatch ... --camera-dir resources/droid/cameras
   --clip-frames 241 --clip-only`) -> `runs/droid/proxies`;
3. link-mode datasets (`python -m simpwm.video.dataset`: links to the raw ext1 MP4 and the RGB / depth proxies at 1280x704)
   -> `runs/droid/datasets/{train,val}`;
4. 241-frame latents (`latents.sbatch`: video, text, RGB and depth proxies through LTX-2's `process_dataset.py`) ->
   `runs/droid/latents241`, and the 97-frame latents sliced from them (`python -m simpwm.video.slice_latents`; the video VAE
   is causal) -> `runs/droid/latents97`;
5. robot-only references (`python -m simpwm.video.robot_refs`, `latents_ref.sbatch`) -> `runs/droid/latents97_robot`.

### Latents

LTX-2's `scripts/process_dataset.py` (in `deps/LTX-2/packages/ltx-trainer`, with `deps/LTX-2/.venv`) encodes the
video, the caption and the reference of each row; `scripts/droid/latents.sbatch` and `scripts/gr1/latents.sbatch` run it per shard:

```bash
python scripts/process_dataset.py DS/dataset.jsonl --resolution-buckets 1280x704x241 \
    --model-path models/ltx-2.3/ltx-2.3-22b-dev.safetensors --text-encoder-path models/gemma-3-12b-it-qat-q4_0-unquantized \
    --reference-downscale-factor 2 --reshape-mode stretch --fps 15 --skip-audio --output-dir PRE/rgb
```

The trainer's data root holds `latents/`, `conditions/` (text) and one directory per reference stream (`ref_rgb/`,
`ref_depth/`, `ref_robot/`, ...): a further stream is encoded from `dataset_<kind>.jsonl` with `latents/` and
`conditions/` linked (`scripts/droid/latents_ref.sbatch`). The video VAE is causal in time, so the first
`(N - 1) // 8 + 1` latent frames of a 241-frame encoding are the encoding of its first N frames:
`python -m simpwm.video.slice_latents SRC_ROOT DST_ROOT latents ref_rgb ref_depth --clip-frames 97` derives the 97-frame
data root without re-encoding.

### Config and training

`python -m simpwm.video.make_config DATA_ROOT VAL_DIR OUTPUT_DIR OUT_YAML IDX ...` fills the template
`configs/ltx/ic_lora.yaml` (LoRA rank 32 on the attention and feed-forward projections, first-frame condition with
probability 0.9, AdamW lr 2e-4 with linear decay, batch 1 per GPU, bf16, gradient checkpointing; validation with 30
steps, CFG 3, STG 1 on block 28, rescale 0.7, seed 42) with the model paths, the reference streams `--refs` (each a
latents directory under DATA_ROOT and a video directory for the validation samples: `control`, `rgb`, `depth`, `rgb,depth`,
`robot`, `pw`, `none`), the reference downscale factor (`--ref-down`, default 2), the validation size / length / frame
rate (`--size`, `--clip-frames`, `--fps`) and the training steps (`--steps`). `--preset gr1` gives the GR1 model.

For the RGB-proxy model of the paper:

```bash
D=runs/droid
python -m simpwm.video.make_config $D/latents97/data $D/datasets/val $D/train/rgb $D/train/rgb.yaml \
    @resources/droid/splits/val.txt --refs rgb --size 1280x704 --clip-frames 97 --fps 15
sbatch -t 2-00:00:00 scripts/droid/train.sbatch $D/train/rgb.yaml configs/accelerate/ddp_8gpu.yaml
```

`train.sbatch` runs the patched trainer with an accelerate config from `configs/accelerate/`:

```bash
cd deps/LTX-2/packages/ltx-trainer
../../.venv/bin/accelerate launch --config_file ../../../../configs/accelerate/ddp_8gpu.yaml scripts/train.py CONFIG_YAML
```

`scripts/droid/train_multinode.sbatch` with `ddp_{2,4}node_4gpu.yaml` runs it over several nodes. Checkpoints: `OUTPUT_DIR/checkpoints/lora_weights_step_XXXXX.safetensors`.

The DROID models of the paper (97 frames, 1280x704, 15 fps, data root of the 97-frame latents, evaluated at step 2000):

| model | `--refs` | data root | steps |
|---|---|---|---|
| RGB proxy (SimPWM) | `rgb` | `latents97` | 3000-step schedule, checkpoint 2000 |
| depth proxy | `depth` | `latents97` | 2000 |
| RGB + depth proxy | `rgb,depth` | `latents97` | 2000 |
| robot only | `robot` | `latents97_robot` | 2000 |
| PointWorld proxy | `pw` | PointWorld proxies encoded as `ref_pw` | 2000 |
| no proxy | `none` | `latents97` | 2000 |

Memory and speed on A100 80 GB GPUs (batch 8 = 8 GPUs x 1):

| references | frames | parallelism | peak memory | s / step |
|---|---|---|---|---|
| RGB proxy | 97 | DDP-8 (`ddp_8gpu.yaml`) | 62.6 GB | ~13 |
| depth proxy | 97 | DDP-8 | ~62.6 GB | ~13 |
| RGB + depth | 97 | DDP-8 | 65.3 GB | ~17 |
| RGB + depth | 241 | FSDP-8 (`fsdp_8gpu.yaml`) | 81.1 GB (at the limit) | ~65 |

241 frames with DDP run out of memory (each process holds the whole 22B model); FSDP shards it over the 8 GPUs.

## 3. MiniMax-H3 LoRA (SimPWM-H3)

The proxy is the reference video `<Video 1>`, the real first frame the reference image `<Picture 1>`, and the first
latent frame of the output is pinned to the first frame. The code is in `simpwm/h3/`.

| | |
|---|---|
| Output | 1280x704, 158 frames at 24 fps (the 97-step, 15 fps clip resampled); `<idx>_15fps.mp4` is the 97-step grid |
| Proxy input | the RGB proxy `ref_rgb_1280x704.mp4`, downscaled to 640x352 |
| Text | a fixed instruction (`PROMPT_HEAD_FOLLOW` in `simpwm/h3/common.py`) followed by the dataset caption |
| LoRA | rank 32 on the attention and feed-forward projections of the 50 blocks (166M parameters, 665 MB) |

Simulate the 845 training episodes with the high-resolution preset, pair them, encode them and train:

```bash
D=runs/h3 IDS=resources/droid/splits/train_h3.txt
# 1. proxies (about 3 min per episode)
sbatch --wait -a 0-31 scripts/droid/proxy.sbatch $IDS 32 $D/proxies --hires \
    --camera-dir resources/droid/cameras_hires --clip-frames 97 --clip-only
# 2. paired dataset: links to the raw video, the RGB proxy and the first frame
python -m simpwm.video.dataset $D/proxies $D/ds @$IDS --link --size 1280x704 --clip-frames 97
# 3. encodings: VAE latents of the proxy, the first frame and the real video, and the text conditioning
sbatch --wait -a 0-23 scripts/h3/encode.sbatch $D/ds $D/cache 24 --ids $IDS
# 4. training
sbatch -N 8 --gres=gpu:4 scripts/h3/train.sbatch $D/ds $D/cache $D/train \
    --ids $IDS --steps 500 --stop-at 250 --lr 1.4e-4 --warmup 25 --save-every 125 --act-offload off
```

Each encoding task runs `python -m simpwm.h3.preencode --stage vae` and `--stage text` on one GPU and writes
`cache/<idx>.vae.safetensors` and `cache/<idx>.text.safetensors`.

The released model is step 250 of this run: 32 GPUs with one sample each (global batch 32), a 500-step linear schedule
stopped at step 250, AdamW, rank 32. On 8 nodes of 4 A100 80 GB GPUs a step takes about 140 s and 66 GB per GPU. The
job resumes from the newest `lora_stepNNNNN.safetensors` in the output directory. With fewer GPUs, `--accum` keeps the
global batch and `--act-offload on` moves the checkpointed activations to the CPU.

| Module | |
|---|---|
| `simpwm/h3/common.py` | geometry, prompt, dataset items, VAE and sequence-layout helpers |
| `simpwm/h3/model.py` | FSDP-sharded transformer with the LoRA |
| `simpwm/h3/seqpar.py` | optional sequence parallelism over several GPUs per sample (`--sp`) |
| `simpwm/h3/preencode.py`, `train.py`, `sample.py` | encodings, training, generation |

## 4. Using the trained model

[inference.md](inference.md) generates videos with either model; [evaluation.md](evaluation.md) reproduces the tables.
