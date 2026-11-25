"""Shorter-clip LTX latents sliced from longer ones, without re-encoding.

The LTX-2 video VAE is causal in time (latent frame 0 = pixel frame 0, then one latent frame per 8 pixel frames), so the
first (N - 1) // 8 + 1 latent frames of a 241-frame encoding are the encoding of its first N frames (on DROID clips
within bf16 noise of a direct N-frame encoding, 0.1-0.2%). The video and reference latents are sliced, the text
conditions are linked; only samples present in every directory are kept (a shard that failed half-way leaves gaps).

usage: python -m simpwm.video.slice_latents SRC_DATA_ROOT DST_DATA_ROOT [DIR ...] [--clip-frames N]
  DATA_ROOT  the trainer's preprocessed_data_root (latents/, conditions/, ref_rgb/, ref_depth/, ...)
  DIR        directories to slice (default: every subdirectory except conditions/)
"""

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from simpwm.video.common import parser

CLIP_FRAMES = 97
WORKERS = 16


def _slice(args):
    src, dst, f = args
    if dst.exists():
        return
    x = torch.load(src, weights_only=False)
    x["latents"] = x["latents"][:, :f].clone()
    x["num_frames"] = f
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    torch.save(x, tmp)
    os.replace(tmp, dst)


def build(src, dst, clip_frames=CLIP_FRAMES, dirs=()):
    src, dst = Path(src).resolve(), Path(dst)
    f = (clip_frames - 1) // 8 + 1
    dst.mkdir(parents=True, exist_ok=True)
    if not (dst / "conditions").exists():
        (dst / "conditions").symlink_to((src / "conditions").resolve())
    dirs = dirs or [d.name for d in src.iterdir() if d.is_dir() and d.name != "conditions"]
    rel = set.intersection(*[{p.relative_to(src / d) for p in (src / d).rglob("*.pt")} for d in [*dirs, "conditions"]])
    jobs = [(src / d / r, dst / d / r, f) for d in dirs for r in sorted(rel)]
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(_slice, jobs, chunksize=32))
    print(f"{len(jobs)} latents -> {dst} ({f} latent frames)")


def main():
    p = parser(__doc__)
    p.add_argument("src", help="data root of the longer latents")
    p.add_argument("dst", help="data root to write")
    p.add_argument("dirs", nargs="*", metavar="DIR", help="latent directories to slice")
    p.add_argument("--clip-frames", type=int, default=CLIP_FRAMES, help="target clip length (default %(default)s)")
    a = p.parse_args()
    build(a.src, a.dst, a.clip_frames, a.dirs)


if __name__ == "__main__":
    main()
