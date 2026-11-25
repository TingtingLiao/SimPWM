# Installation

## 1. Environment

```bash
git clone https://github.com/TingtingLiao/SimPWM.git && cd SimPWM
conda create -n simpwm python=3.10 -y && conda activate simpwm
pip install -e ".[raster,eval]"
```

You need a CUDA GPU. The extras are optional:

| Extra | Installs | Needed for |
|:---|:---|:---|
| `raster` | [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | GPU robot rasterizer; it compiles on first use and needs a CUDA toolkit in `CUDA_HOME`. Without it, the robot is ray-cast with MuJoCo on the CPU. |
| `eval` | lpips, piq, torchvision | the evaluation metrics |
| `gr1` | pyarrow | the GR1 dataset |
| `droid-rlds` | tfrecord | the RLDS release of DROID (the raw release needs nothing extra) |
| `h3` | diffusers (development version) | the MiniMax-H3 video model; install it in its own environment (section 4) |

## 2. Robot models

```bash
git clone --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie assets/mujoco_menagerie
git -C assets/mujoco_menagerie sparse-checkout set franka_fr3 robotiq_2f85
```

The GR1 robot model is described in [gr1.md](gr1.md).

## 3. Perception models

The models are loaded from the Hugging Face Hub (`HF_HOME`). The Slurm launchers set `HF_HUB_OFFLINE=1`, so download
them first with `hf download <model>`:

| Model | Used for |
|:---|:---|
| `IDEA-Research/grounding-dino-base` | object proposals |
| `facebook/sam2.1-hiera-large` | masks, tracks in the real video |
| `facebook/sam3` | concept proposals, tracks (or a local copy in `SAM3_PATH`) |
| `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | metric depth |
| `Qwen/Qwen3-VL-32B-Instruct` | naming deformable objects |

## 4. Video models

Needed for training ([train.md](train.md)) and generation ([inference.md](inference.md)), not for the simulator.

### LTX-2.3

| what | where | notes |
|---|---|---|
| LTX-2 trainer | `deps/LTX-2` | `git clone https://github.com/Lightricks/LTX-2 deps/LTX-2 && git -C deps/LTX-2 checkout e98f6d927d83c92ef2be6aec76c8440b19306e39` |
| our trainer patch | `patches/ltx_trainer.patch` | `git -C deps/LTX-2 apply ../../patches/ltx_trainer.patch` |
| trainer environment | `deps/LTX-2/.venv` | `cd deps/LTX-2 && uv sync --extra natten` (PyTorch 2.13 / CUDA 13: NVIDIA driver >= 580) |
| LTX-2.3 checkpoint | `models/ltx-2.3/ltx-2.3-22b-dev.safetensors` | the 22B dev checkpoint of the LTX-2 release |
| text encoder | `models/gemma-3-12b-it-qat-q4_0-unquantized` | Hugging Face `google/gemma-3-12b-it-qat-q4_0-unquantized` |

The patch changes the trainer as follows; its environment variables are read by the trainer processes:

| change | file |
|---|---|
| `--fps F`: the frame rate recorded for every video instead of its container's (DROID's raw MP4s are muxed at 60 fps for a 15 Hz recording); `--reshape-mode stretch`: resize to the bucket without cropping (720 -> 704 rows) | `scripts/process_dataset.py`, `process_videos.py` |
| LR schedule length x `num_processes` (accelerate steps a prepared scheduler once per process; without it a multi-GPU run finishes its schedule early) | `trainer.py` |
| validation sampling with the DDP-unwrapped model (FSDP keeps its wrapper) | `trainer.py` |
| `SIMPWM_BUCKET_MAP=<dataset.jsonl>`: length-bucketed data, every rank draws the same bucket at each step | `trainer.py` |
| `SIMPWM_DIST_TIMEOUT_S=<seconds>`: process-group timeout (long validations) | `trainer.py` |
| `SIMPWM_REF_POS_FIX=1`: the reference scale factors are inferred under the data root, so reference positions are rescaled in training as at inference (GR1 model) | `training_strategies/flexible.py` |

### MiniMax-H3

H3 needs its own environment, because it uses a development version of diffusers:

```bash
conda create -n simpwm-h3 python=3.11 -y && conda activate simpwm-h3
pip install -e ".[h3]"          # diffusers at the commit that added MiniMax-H3
export H3_PY=$(which python)    # read by scripts/h3/*.sbatch
```

Download the base model and convert its Ref2VA transformer to the diffusers layout (CPU, about 200 GB of memory):

```bash
hf download MiniMaxAI/MiniMax-H3 --local-dir models/MiniMax-H3
git clone https://github.com/huggingface/diffusers deps/diffusers
git -C deps/diffusers checkout f53d552036a0d1bd5570782a39cd40cfabf112bc
M=models/MiniMax-H3 O=models/MiniMax-H3-diffusers-ref2va
python deps/diffusers/scripts/convert_minimax_h3_to_diffusers.py --checkpoint_path $M/Ref2VA --output_path $O
for d in text_encoder tokenizer processor; do mkdir -p $O/$d && ln -sf $(realpath $M/Ref2VA/$d)/* $O/$d/; done
```

`H3_DIFFUSERS` overrides the location of the converted model (default `models/MiniMax-H3-diffusers-ref2va`). The base
model has its own license (MiniMax H3 Community License).

## 5. Launch scripts

The Slurm launchers in `scripts/` source `scripts/env.sh`, which reads `PY` (the Python of this environment),
`CUDA_HOME`, `SIMPWM_DEPS` and `SIMPWM_RUNS`; `HF_HOME` and `SAM3_PATH` are read by the Python code. Submit them from the repository root.
