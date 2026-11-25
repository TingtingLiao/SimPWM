<div align="center">

<h2>SimPWM: Simulator-as-Proxy World Model for Robot Manipulation</h2>

<p>
  <a href="https://tingtingliao.github.io/"><b>Tingting&nbsp;Liao</b></a><sup>1</sup> &nbsp;·&nbsp;
  <a href="https://mignonjia.github.io/"><b>Mingjia&nbsp;Huo</b></a><sup>1,2</sup> &nbsp;·&nbsp;
  <a href="https://www.hao-li.com/"><b>Hao&nbsp;Li</b></a><sup>1,3,*</sup> &nbsp;·&nbsp;
  <a href="https://hunterhector.github.io/"><b>Zhengzhong&nbsp;Liu</b></a><sup>1</sup> &nbsp;·&nbsp;
  <a href="https://www.cs.cmu.edu/~epxing/"><b>Eric&nbsp;P.&nbsp;Xing</b></a><sup>1</sup> &nbsp;·&nbsp;
  <a href="https://guangyliu.github.io/"><b>Guangyi&nbsp;Liu</b></a><sup>1</sup>
  <br>
  <sub><sup>1</sup>IFM, MBZUAI &nbsp;&nbsp; <sup>2</sup>UC San Diego &nbsp;&nbsp; <sup>3</sup>Pinscreen &nbsp;&nbsp; <sup>*</sup>Corresponding author</sub>
</p>

<a href='https://arxiv.org/abs/2611.32786'><img src='https://img.shields.io/badge/arXiv-2611.32786-b31b1b?logo=arxiv&logoColor=b31b1b&labelColor=white'></a> &ensp;
<a href='https://sim-pwm.pages.dev/'><img src='https://img.shields.io/badge/project-page-green?logo=googlechrome&logoColor=green&labelColor=white'></a> &ensp;
<a href='https://huggingface.co/Luffuly/SimPWM-H3-C'><img src='docs/media/badge_weights.svg'></a> &ensp;
<a href='LICENSE'><img src='https://img.shields.io/badge/license-MIT-blue?labelColor=white'></a>

</div>

https://github.com/user-attachments/assets/774cae88-13b2-46fd-9b36-c976b7f806cc

<p align="center"><sub><b>Real2Sim:</b> the simulator's 4D state of the scene and the robot. <b>Sim2Real:</b> the SimPWM video rendered from it.</sub></p>

---

**A rule-based simulator decides *what moves*; a video model renders *how it looks*.** Video world models often lose the object the robot handles, and simulators alone look synthetic. SimPWM uses the simulator as a proxy and gets both right.

## 🔧 Installation

```bash
git clone https://github.com/TingtingLiao/SimPWM.git && cd SimPWM
conda create -n simpwm python=3.10 -y && conda activate simpwm
pip install -e ".[raster,eval]"

# robot models (Franka FR3, Robotiq 2F-85)
git clone --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie assets/mujoco_menagerie
git -C assets/mujoco_menagerie sparse-checkout set franka_fr3 robotiq_2f85
```

> Needs a CUDA GPU. See the [installation guide](docs/installation.md) for the optional extras and the perception models.

## 🚀 Quick Start

Run the simulator on a bundled example. Each example holds a first frame, a calibrated camera and an action sequence, so no dataset is needed:

```bash
python -m simpwm.demo examples/bottle
```

| Example | Task |
|:---|:---|
| [`examples/bottle`](examples/bottle) | Put the bottle on top of the microwave |
| [`examples/rope`](examples/rope) | Clump objects (cloth, rope, plush toy) |
| [`examples/shirt`](examples/shirt) | Put the black t-shirt on the black stand |

```
runs/demo/bottle/
├── preview.mp4        first frame | RGB proxy | depth proxy
├── proxy_rgb.mp4      RGB proxy (input of the video model)
├── proxy_depth.mp4    depth proxy
└── info.json          camera, initial state, simulator settings
```

One example takes about 3 minutes on one GPU. To simulate your own clip, make a folder with the same files (a 1280x720 first frame); the formats are described in [`simpwm/demo.py`](simpwm/demo.py).

Then render the proxies into video with SimPWM-H3. This needs the H3 environment, the base model and our LoRA ([setup](docs/installation.md#4-video-models)), and 4 GPUs of 80 GB:

```bash
hf download Luffuly/SimPWM-H3-C --local-dir models/simpwm-h3
bash scripts/h3/demo.sh runs/demo        # -> runs/demo/h3c/bottle.mp4
```

## 📘 Instructions

- See [installation](docs/installation.md) to set up the environment, the robot models, the perception models and the video models
- See [dataset](docs/dataset.md) to prepare DROID and its stereo depth, and to run the simulator on the train and test episodes
- See [train](docs/train.md) to train the LTX-2.3 IC-LoRA and the MiniMax-H3 LoRA
- See [inference](docs/inference.md) to download the SimPWM-H3 weights and render proxies into video
- See [evaluation](docs/evaluation.md) to compute the metrics of the paper
- See [configuration](docs/configuration.md) for the simulator settings, the data paths and the shipped splits

## 🙏 Acknowledgments

Here are some great resources we benefit from:

- [LTX-2](https://github.com/Lightricks/LTX-2) and [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) for the video model
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [SAM 2](https://github.com/facebookresearch/sam2) and [SAM 3](https://github.com/facebookresearch/sam3) for object detection, segmentation and tracking
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and [FoundationStereo](https://github.com/NVlabs/FoundationStereo) for monocular and stereo depth
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) for the robot models, [nvdiffrast](https://github.com/NVlabs/nvdiffrast) for rasterization
- [DROID](https://droid-dataset.github.io), [GR00T-Teleop](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1) and [WorldArena](https://github.com/tsinghua-fib-lab/WorldArena) for datasets and benchmarks
- [DreamDojo](https://github.com/NVIDIA/DreamDojo) for the GR1 comparison

## 📖 Citation

```bibtex
@misc{liao2026simpwm,
  title         = {Simulator-as-Proxy World Model for Robot Manipulation},
  author        = {Tingting Liao and Mingjia Huo and Hao Li and Zhengzhong Liu and Eric P. Xing and Guangyi Liu},
  year          = {2026},
  eprint        = {2611.32786},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2611.32786}
}
```

## 📄 License

[MIT](LICENSE). External models and datasets keep their own licenses.
