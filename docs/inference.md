# Inference

Rendering proxies into video with a trained model. Both models need their environment and base weights
([installation.md](installation.md#4-video-models)).

## 1. Weights

| Model | Training data | Checkpoint |
|---|---|---|
| SimPWM-H3 (MiniMax-H3 LoRA, 1280x704) | 845 DROID episodes, `resources/droid/splits/train_h3.txt` | `hf download Luffuly/SimPWM-H3-C --local-dir models/simpwm-h3` |

The LTX-2.3 models of the paper are trained with [train.md](train.md#2-ltx-23-ic-lora).

## 2. Bundled examples

Run the quick start, then `scripts/h3/demo.sh`, which encodes and generates every example of `runs/demo` on the 4 GPUs
of the current node:

```bash
python -m simpwm.demo examples/bottle examples/rope examples/shirt    # main environment
bash scripts/h3/demo.sh runs/demo                                     # -> runs/demo/h3c/<name>.mp4
```

## 3. MiniMax-H3

The model reads pre-encoded inputs. Simulate the episodes with `--hires` ([dataset.md](dataset.md#4-proxies)), build a
paired dataset with `python -m simpwm.video.dataset RUN DS @IDS --link --size 1280x704 --clip-frames 97`
([train.md](train.md#1-paired-datasets)), encode it without the real video and generate:

```bash
sbatch -a 0-7 scripts/h3/encode.sbatch DS CACHE 8 --no-target
sbatch scripts/h3/generate.sbatch CACHE OUT h3c models/simpwm-h3/simpwm_h3c_1280x704.safetensors IDS.txt
```

This runs `torchrun --standalone --nproc-per-node 4 -m simpwm.h3.sample`. The frozen model is sharded over the GPUs of
the node (FSDP), and each GPU generates its own video: 50-point schedule, video shift 12, seed 0. On 4 A100 80 GB GPUs
a video takes about 27 minutes and 42 GB per GPU. `LORA = none` gives the zero-shot base model.

Output: `OUT/h3c/<idx>.mp4`, 1280x704, 158 frames at 24 fps (the 97-step, 15 fps clip resampled), and
`<idx>_15fps.mp4` on the 97-step grid.

## 4. LTX-2.3

```bash
sbatch -a 0-19 scripts/droid/generate.sbatch CONFIG_YAML LORA runs/droid/datasets/test OUT_DIR 20
```

`generate.sbatch` renders every episode of a paired dataset (the test set of [dataset.md](dataset.md#5-test-set)) with
one shard per GPU. The videos are rendered through the trainer's own validation pipeline, so the inference settings
are exactly those of training-time validation:

```bash
python -m simpwm.video.eval_config TRAIN_CONFIG LORA EVAL_DIR OUT/shard_0 --shard 0 4   # -> cfg.yaml, order.json
cd deps/LTX-2/packages/ltx-trainer
../../.venv/bin/accelerate launch --config_file ../../../../configs/accelerate/single_gpu.yaml scripts/train.py OUT/shard_0/cfg.yaml
```

The LoRA is loaded without its optimizer state, the learning rate is 0, and the step-0 validation renders one sample
per episode: its reference video(s) (the same streams and downscale factor as the training config's validation
samples), its real first frame and its caption. Sample `k` (1-based) is `OUT/shard_0/samples/step_000000_<k>.mp4`, a
panel [reference | generation] of the `k`-th episode of `order.json`; `python -m simpwm.eval.wm_bench export OUT VIDEOS`
cuts the generated half into `VIDEOS/<idx>.mp4`.

The zero-training baseline input (`python -m simpwm.video.prep`) writes a per-episode grayscale depth proxy, the first
frame, the real video and the prompt for the pretrained IC-LoRA pipeline.
