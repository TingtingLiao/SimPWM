"""LTX-2 IC-LoRA training config from the template configs/ltx/ic_lora.yaml: model paths, data root, output directory,
the reference conditions (--refs, in this order) and one validation sample per validation episode IDX of VAL_DIR (its
reference video(s) downscaled by --ref-down, its real first frame, its caption).

  --refs  comma list of reference streams; each is a latents directory under DATA_ROOT and a video directory of the
          video dataset: control (reference_latents/, refs/; dataset.py default mode), rgb (ref_rgb/, refs/; RGB
          proxy of a --link dataset), depth (ref_depth/, refs_depth/), robot (ref_robot/, refs_robot/; robot_refs.py),
          pw (ref_pw/, refs_pw/); "none": first frame and text only
  --preset gr1  the GR1 model: rgb reference at full resolution, 800x480x97 at 10 fps, 2000 steps at lr 2e-4 (linear),
          checkpoints and validation every 500 steps (keep the last 2), no step-0 validation
Without --size / --clip-frames / --fps / --steps the template's values are kept (768x448x97, 24 fps, 3000 steps).

usage: python -m simpwm.video.make_config DATA_ROOT VAL_DIR OUTPUT_DIR OUT_YAML IDX [IDX ...] [--refs LIST]
           [--ref-down K] [--size WxH] [--clip-frames N] [--fps F] [--steps N] [--preset gr1] [--model F]
           [--text-encoder D]
  DATA_ROOT  the trainer's preprocessed_data_root (latents/, conditions/ and the reference latents directories)
  VAL_DIR    a video dataset (refs*/, first/, dataset.jsonl) holding the validation episodes
"""

import json
from pathlib import Path

import yaml

from simpwm import paths
from simpwm.video.common import parser, size_arg

MODELS = paths.REPO / "models"
MODEL = MODELS / "ltx-2.3/ltx-2.3-22b-dev.safetensors"
GEMMA = MODELS / "gemma-3-12b-it-qat-q4_0-unquantized"
TEMPLATE = paths.REPO / "configs/ltx/ic_lora.yaml"
REFS = {  # reference stream -> (latents directory under the data root, video directory of the video dataset)
    "control": ("reference_latents", "refs"), "rgb": ("ref_rgb", "refs"), "depth": ("ref_depth", "refs_depth"),
    "robot": ("ref_robot", "refs_robot"), "pw": ("ref_pw", "refs_pw"),
}
GR1 = dict(refs=["rgb"], ref_down=1, size=(800, 480), clip_frames=97, fps=10.0)


def template(data_root, out_dir, model=MODEL, text_encoder=GEMMA):
    """The template config with the model paths, the data root and the output directory filled in."""
    cfg = yaml.safe_load(TEMPLATE.read_text())
    cfg["model"]["model_path"], cfg["model"]["text_encoder_path"] = str(model), str(text_encoder)
    # absolute: the trainer resolves relative paths against its own directory
    cfg["data"]["preprocessed_data_root"], cfg["output_dir"] = str(Path(data_root).resolve()), str(Path(out_dir).resolve())
    return cfg


def samples(val_dir, idxs, ref_dirs, down=2):
    """One validation sample per episode: its reference video(s) VAL_DIR/<ref_dir>/<idx>.mp4 (downscaled by `down`),
    its real first frame, its caption."""
    val = Path(val_dir).resolve()
    rows = {r["idx"]: r for r in map(json.loads, open(val / "dataset.jsonl"))}
    return [dict(prompt=rows[i]["caption"],
                 conditions=[dict(type="reference", video=str(val / d / f"{i}.mp4"), downscale_factor=down,
                                  temporal_scale_factor=1, include_in_output=True) for d in ref_dirs]
                 + [dict(type="first_frame", image_or_video=str(val / "first" / f"{i}.png"))])
            for i in idxs]


def build(data_root, val_dir, out_dir, idxs, refs=("control",), ref_down=2, size=None, clip_frames=None, fps=None,
          steps=None, model=MODEL, text_encoder=GEMMA):
    """The training config (dict); size / clip_frames / fps / steps None keep the template's."""
    cfg = template(data_root, out_dir, model, text_encoder)
    cfg["validation"]["samples"] = samples(val_dir, idxs, [REFS[r][1] for r in refs], ref_down)
    conds = [c for c in cfg["training_strategy"]["video"]["conditions"] if c["type"] != "reference"]
    cfg["training_strategy"]["video"]["conditions"] = [dict(type="reference", latents_dir=REFS[r][0], probability=1.0)
                                                       for r in refs] + conds
    v = cfg["validation"]
    if size or clip_frames:
        w, h = size or v["video_dims"][:2]
        v["video_dims"] = [w, h, clip_frames or v["video_dims"][2]]
    if fps:
        v["frame_rate"] = float(fps)
    if steps:
        cfg["optimization"]["steps"] = steps
    return cfg


def gr1(cfg):
    """The GR1 training schedule on top of build(**GR1)."""
    cfg["optimization"].update(steps=2000, learning_rate=2e-4, scheduler_type="linear", batch_size=1,
                               gradient_accumulation_steps=1)
    cfg["checkpoints"].update(interval=500, keep_last_n=2)
    cfg["validation"].update(interval=500, skip_initial_validation=True, video_dims=[800, 480, 97], frame_rate=10.0)
    assert all(c["downscale_factor"] == 1 for s in cfg["validation"]["samples"] for c in s["conditions"]
               if c["type"] == "reference")
    assert [c for c in cfg["training_strategy"]["video"]["conditions"]
            if c["type"] == "first_frame"][0]["probability"] == 0.9
    return cfg


def main():
    p = parser(__doc__)
    p.add_argument("data_root", help="preprocessed data root of the training latents")
    p.add_argument("val_dir", help="video dataset of the validation episodes")
    p.add_argument("output_dir", help="training output directory (checkpoints, validation samples)")
    p.add_argument("out_yaml", help="config file to write")
    p.add_argument("idxs", nargs="+", type=int, metavar="IDX", help="validation episodes")
    p.add_argument("--preset", choices=["gr1"], help="dataset preset (defaults of the options below + schedule)")
    p.add_argument("--refs", help=f"comma list of {', '.join(REFS)} or none (default: control; gr1: rgb)")
    p.add_argument("--ref-down", type=int, help="reference downscale factor (default 2; gr1: 1)")
    p.add_argument("--size", type=size_arg, metavar="WxH", help="validation size (default: the template's)")
    p.add_argument("--clip-frames", type=int, help="validation clip length (default: the template's, 97)")
    p.add_argument("--fps", type=float, help="validation frame rate (default: the template's)")
    p.add_argument("--steps", type=int, help="training steps (default: the template's)")
    p.add_argument("--model", type=Path, default=MODEL, help="LTX-2 checkpoint (default %(default)s)")
    p.add_argument("--text-encoder", type=Path, default=GEMMA, help="Gemma text encoder (default %(default)s)")
    a = p.parse_args()
    opts = dict(GR1) if a.preset == "gr1" else dict(refs=["control"], ref_down=2)
    if a.refs is not None:
        opts["refs"] = [] if a.refs == "none" else a.refs.split(",")
    for k in ("ref_down", "size", "clip_frames", "fps", "steps"):
        if getattr(a, k) is not None:
            opts[k] = getattr(a, k)
    cfg = build(a.data_root, a.val_dir, a.output_dir, a.idxs, model=a.model, text_encoder=a.text_encoder, **opts)
    if a.preset == "gr1":
        cfg = gr1(cfg)
    Path(a.out_yaml).write_text(yaml.safe_dump(cfg, sort_keys=False, width=120))
    print("config ->", a.out_yaml, len(a.idxs), "validation samples")


if __name__ == "__main__":
    main()
