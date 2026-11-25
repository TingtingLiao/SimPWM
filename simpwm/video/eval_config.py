"""Generation config: render episodes of a video dataset with a trained IC-LoRA through the trainer's own validation
pipeline, so the inference settings are exactly those of training-time validation. The LoRA is loaded without resuming
optimizer / step state, the learning rate is 0, and the validation at step 0 renders one sample per episode: its
reference video(s) (the same reference directories and downscale factor as the training config's validation samples),
its real first frame and its caption.

Writes OUT_DIR/cfg.yaml and OUT_DIR/order.json (the episode of each sample: the trainer writes sample k, 1-based, to
OUT_DIR/samples/step_000000_<k>.mp4). Episodes: IDX, or rows T, T + N, ... of EVAL_DIR/dataset.jsonl with --shard T N.

usage: python -m simpwm.video.eval_config TRAIN_CONFIG LORA EVAL_DIR OUT_DIR [IDX ...] [--shard T N] [--clip-frames N]
"""

import json
from pathlib import Path

import yaml

from simpwm.video.common import parser, read_jsonl
from simpwm.video.make_config import samples


def reference_streams(cfg):
    """(reference video directories, downscale factor) of a training config's validation samples."""
    refs = [c for c in cfg["validation"]["samples"][0]["conditions"] if c["type"] == "reference"]
    return [Path(c["video"]).parent.name for c in refs], (refs[0]["downscale_factor"] if refs else 1)


def build(train_cfg, lora, eval_dir, out_dir, idxs, clip_frames=None):
    cfg = yaml.safe_load(Path(train_cfg).read_text())
    ref_dirs, down = reference_streams(cfg)
    cfg["model"]["load_checkpoint"] = str(Path(lora).resolve())
    cfg["checkpoints"].update(no_resume=True, interval=None)
    cfg["optimization"].update(steps=1, learning_rate=0.0)
    cfg["validation"].update(interval=100000, skip_initial_validation=False,
                             samples=samples(eval_dir, idxs, ref_dirs, down))
    if clip_frames:
        cfg["validation"]["video_dims"] = cfg["validation"]["video_dims"][:2] + [clip_frames]
    cfg["output_dir"] = str(Path(out_dir).resolve())
    return cfg


def main():
    p = parser(__doc__)
    p.add_argument("train_cfg", help="the LoRA's training config")
    p.add_argument("lora", help="LoRA checkpoint (lora_weights_step_XXXXX.safetensors)")
    p.add_argument("eval_dir", help="video dataset of the episodes to render (refs*/, first/, dataset.jsonl)")
    p.add_argument("out_dir", help="generation output directory")
    p.add_argument("idxs", nargs="*", type=int, metavar="IDX", help="episodes (default: every row, or --shard's)")
    p.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("T", "N"),
                   help="the T-th of N interleaved shards of the dataset rows")
    p.add_argument("--clip-frames", type=int, help="generated frames (default: the training config's)")
    a = p.parse_args()
    idxs = a.idxs or [r["idx"] for r in read_jsonl(Path(a.eval_dir) / "dataset.jsonl")][a.shard[0]::a.shard[1]]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "order.json").write_text(json.dumps(idxs))
    (out / "cfg.yaml").write_text(yaml.safe_dump(build(a.train_cfg, a.lora, a.eval_dir, out, idxs, a.clip_frames),
                                                 sort_keys=False, width=120))
    print("generation config ->", out / "cfg.yaml", len(idxs), "samples")


if __name__ == "__main__":
    main()
