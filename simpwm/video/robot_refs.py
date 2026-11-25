"""Robot-only reference videos (Table 3's "robot" model): the rendered robot alone, no objects and no scene, next to
the RGB / depth proxies of a --link video dataset (video/dataset.py).

The robot is the G channel of each episode's control video RUN/droid_XXXXX/conditions/ref_1280x704.mp4 (Lambert-shaded
mesh luminance on the robot, 0 elsewhere; values below BLACK, compression bleed from the other channels, are set to 0),
written as a grey video DS/refs_robot/<idx>.mp4 (first --clip-frames frames, same size and frame rate), plus
DS/dataset_robot.jsonl (dataset.jsonl with reference_video = refs_robot/<idx>.mp4) for the latent encoding.

usage: python -m simpwm.video.robot_refs RUN DS [--clip-frames N] [--shard T N] [--force]
"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from simpwm.video.common import parser, read_jsonl, write_jsonl, write_video

BLACK = 16
CLIP_FRAMES = 97
CONTROL = "ref_1280x704.mp4"


def build(run, ds, clip_frames=CLIP_FRAMES, task=0, n_tasks=1, force=False):
    run, ds = Path(run), Path(ds)
    out = ds / "refs_robot"
    out.mkdir(exist_ok=True)
    rows = read_jsonl(ds / "dataset.jsonl")
    if task == 0:
        write_jsonl(ds / "dataset_robot.jsonl", [dict(r, reference_video=f"refs_robot/{r['idx']}.mp4") for r in rows])
    for r in rows[task::n_tasks]:
        dst = out / f"{r['idx']}.mp4"
        if dst.exists() and not force:
            continue
        rd = imageio.get_reader(run / f"droid_{r['idx']:05d}" / "conditions" / CONTROL)
        fps = rd.get_meta_data().get("fps", 15)
        frames = [np.repeat(np.where(f[..., 1:2] >= BLACK, f[..., 1:2], 0), 3, -1)
                  for _, f in zip(range(clip_frames), rd)]
        rd.close()
        write_video(dst, frames, fps)
        print(r["idx"], len(frames), flush=True)


def main():
    p = parser(__doc__)
    p.add_argument("run", help="proxy run directory (droid/run.py output)")
    p.add_argument("ds", help="--link video dataset of that run (dataset.jsonl)")
    p.add_argument("--clip-frames", type=int, default=CLIP_FRAMES, help="frames per video (default %(default)s)")
    p.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("T", "N"),
                   help="the T-th of N interleaved shards of the rows")
    p.add_argument("--force", action="store_true", help="rewrite existing videos")
    a = p.parse_args()
    build(a.run, a.ds, a.clip_frames, *a.shard, a.force)


if __name__ == "__main__":
    main()
