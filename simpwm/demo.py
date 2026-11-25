"""Run the simulator on a self-contained example: first frame + camera + actions -> proxy videos.

An example directory (examples/<name>/) holds
  first_frame.png    the first frame of the camera (1280x720)
  camera.json        the calibrated camera: K, T_wc (camera -> robot base), dist at 320x180, and gripper_yaw
  actions.npz        arm_q (T, 7) joint angles and grip (T,) gripper closure in [0, 1] at `fps`
  stereo_depth.npz   optional: metric depth of the first frame (720x1280, NaN = unknown); monocular depth without it
  prompt.txt         the task

The high-resolution preset (`droid_hires`) simulates the actions and writes to OUT/<name>/
  proxy_rgb.mp4, proxy_depth.mp4   the RGB and depth proxies (the inputs of the video model), W x H, at `fps`
  preview.mp4                      first frame | RGB proxy | depth proxy, side by side
  info.json                        camera, initial state, simulator settings, object states
and adds the example to OUT/dataset.jsonl (caption, proxy, first frame), the input of the video model:
`bash scripts/h3/demo.sh OUT` renders the proxies with SimPWM-H3 (docs/inference.md).

usage: python -m simpwm.demo examples/bottle [--out runs/demo] [--size 1280x704] [--frames N]
"""

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from simpwm import config
from simpwm.core.camera import Camera
from simpwm.core.embodiment import Embodiment
from simpwm.droid.run import label_held, write_control_videos
from simpwm.sim import init_state, pipeline, scale
from simpwm.video.dataset import prompt


def load(example):
    """dict(first (H, W, 3) uint8, arm_q, grip, fps, camera (dict), stereo (H, W) float32 or None, text)."""
    d = Path(example)
    act = np.load(d / "actions.npz")
    stereo = d / "stereo_depth.npz"
    return dict(first=np.asarray(Image.open(d / "first_frame.png").convert("RGB")), arm_q=act["arm_q"], grip=act["grip"],
                fps=float(act["fps"]), camera=json.loads((d / "camera.json").read_text()),
                stereo=np.load(stereo)["depth"].astype(np.float32) if stereo.exists() else None,
                text=(d / "prompt.txt").read_text().strip())


def run(example, out, size="1280x704", frames=None):
    """Simulate the example and write its proxies to out (module docstring)."""
    config.use("droid_hires")
    ex = load(example)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    first = ex["first"]
    h, w = scale.sim_hw()
    frame0 = cv2.resize(first, (w, h), interpolation=cv2.INTER_AREA)  # the simulation frame
    c = ex["camera"]
    cam = Camera(np.array(c["K"]), np.array(c["T_wc"]), 180, 320, tuple(c["dist"])).scaled(h, w)
    emb = Embodiment("franka_robotiq")
    emb.set_gripper_yaw(c["gripper_yaw"])
    T = len(ex["arm_q"])
    steps = np.arange(min(frames or T, T))
    init_state.STEREO = ex["stereo"]
    try:
        res = pipeline.generate(frame0, ex["arm_q"], ex["grip"], cam, emb, ex["fps"], refine_cam=False,
                                hires=(cam.scaled(*first.shape[:2]), set(steps.tolist())), frame0_hi=first)
    finally:
        init_state.STEREO = None
    m0 = {o.id: m for o, m in zip(res["objects0"], res["dbg"]["obj_masks"])}
    held = label_held(res)
    write_control_videos(None, dict(fps=ex["fps"]), res, m0, held, out, size, steps, first)
    (out / f"ref_rgb_{size}.mp4").rename(out / "proxy_rgb.mp4")
    (out / f"ref_depth_{size}.mp4").rename(out / "proxy_depth.mp4")
    (out / f"ref_{size}.mp4").unlink()  # the control encoding (depth / robot / object channels) is not used here
    W, H = (int(x) for x in size.split("x"))
    still = cv2.resize(first, (W, H), interpolation=cv2.INTER_AREA)
    rgb, depth = imageio.get_reader(out / "proxy_rgb.mp4"), imageio.get_reader(out / "proxy_depth.mp4")
    imageio.mimwrite(out / "preview.mp4", [np.concatenate([still, a, b], 1)[::2, ::2] for a, b in zip(rgb, depth)],
                     fps=ex["fps"], quality=8, macro_block_size=1)
    info = dict(res["info"], text=ex["text"], states=res["states"][::5], K=res["cam"].K.tolist(),
                T_wc=res["cam"].T_wc.tolist(), dist=list(res["cam"].dist))
    (out / "info.json").write_text(json.dumps(info, indent=1, default=float))
    Image.fromarray(first).save(out / "first_frame.png")
    index = out.parent / "dataset.jsonl"  # the paired-dataset layout read by simpwm.h3.preencode
    rows = [json.loads(r) for r in index.read_text().splitlines()] if index.exists() else []
    row = dict(idx=out.name, caption=prompt(ex["text"]), reference_video=f"{out.name}/proxy_rgb.mp4",
               first=f"{out.name}/first_frame.png")
    index.write_text("".join(json.dumps(r) + "\n" for r in [r for r in rows if r["idx"] != out.name] + [row]))
    print("done", out, "| held objects:", held, flush=True)


def main():
    ap = argparse.ArgumentParser(prog="python -m simpwm.demo", description=__doc__.split("\n\n")[0])
    ap.add_argument("example", type=Path, nargs="+", help="example directories (examples/<name>)")
    ap.add_argument("--out", type=Path, default=Path("runs/demo"), help="output root (one directory per example)")
    ap.add_argument("--size", default="1280x704", metavar="WxH", help="size of the proxy videos")
    ap.add_argument("--frames", type=int, help="render only the first N steps (default: all)")
    a = ap.parse_args()
    for ex in a.example:
        run(ex, a.out / ex.name, a.size, a.frames)


if __name__ == "__main__":
    main()
