"""Paired (proxy video -> real video) LTX IC-LoRA dataset for DROID from droid/run.py proxy runs, and the encodings of
the proxy videos.

Proxy encodings (fixed ranges, so a value means the same in every episode):
  control_frame    R = inverse depth over 0.2-3 m (near = bright; beyond 3 m or no depth = 0, which drops the noisy far
                   background), G = robot (optionally Lambert-shaded), B = the held object(s)
  world_rgb_frame  RGB proxy: the first frame as the static scene, the robot mesh and held objects drawn over it, their
                   uncovered first-frame footprint in MARK_RGB ("unknown, to be completed")
  depth_frame      depth proxy: the inverse depth of control_frame's R channel as a grey image

Per episode IDX of RUN, in OUT (ltx-trainer convention):
  refs/<idx>.mp4    the proxy                     videos/<idx>.mp4  the real ext1 video on the same time axis
  first/<idx>.png   the real first frame          dataset.jsonl     {caption, video, reference_video, idx} rows
Modes:
  default  the control video from conditions/ (24 fps), the frames of the raw ext1 MP4 resampled to 24 fps, both at
           --size; skipped unless the MP4 is the loader's
  --raw    the full-resolution raw ext1 MP4 at its own 15 fps (the first --clip-frames steps), the proxy
           conditions/ref_WxH.mp4 rendered at that resolution by droid/run.py; skipped unless the MP4 is the loader's
  --link   (raw DROID source; the paper's setup) no re-encoding: videos/ links the raw ext1 MP4, refs/ the RGB proxy
           conditions/ref_rgb_WxH.mp4, refs_depth/ the depth proxy ref_depth_WxH.mp4; dataset_depth.jsonl pairs the
           videos with the depth proxy. The trainer's preprocessing takes the first frames and stretches 720 -> 704
           (process_dataset.py --reshape-mode stretch --fps 15).

usage: python -m simpwm.video.dataset RUN OUT IDX [IDX ...] [--link] [--raw] [--size WxH] [--clip-frames N]
"""

import glob
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from simpwm import paths
from simpwm.droid import data as droid_data
from simpwm.droid.run import CWM_FPS, CWM_HW
from simpwm.sim.render import C_ARM, CWM_MAP, CWM_OBJ
from simpwm.video.common import caption, parser, read_frames, size_arg, write_jsonl, write_video

SIZE = (768, 448)  # default output size (W, H)
CLIP_FRAMES = 97  # clip length in frames (8k + 1), from the first recorded step
STEREO_DIR = paths.DATA / "droid_raw_stereo"  # droid/stereo/download.py output: index.json + per-episode recordings
RAW_MATCH_MAX = 25.0  # mean abs difference (0-255) of the raw and loader first frames at the loader's size
RAW_MATCH_PATCH_MAX = 12.0  # raw_first_frame: worst 16x16 block difference allowed
D_NEAR, D_FAR = 0.2, 3.0
ROBOT = CWM_MAP[C_ARM]
SHADE_FLOOR = 0.25  # keeps the darkest robot parts (black gripper) apart from the background (0)
MARK_RGB = (255, 0, 255)  # "unknown, to be completed": the first-frame footprint of what has moved off it
SCENE = "A static camera films a Franka robot arm with a two-finger gripper working on a table in a lab."


def prompt(text):
    """The DROID caption of a task instruction (its first line)."""
    return caption(SCENE, (text or "").strip().split("\n")[0].rstrip("."), "The robot arm")


def _inverse_depth(depth):
    """Inverse depth scaled to [0, 1] over D_NEAR-D_FAR (near = 1); 0 beyond D_FAR or without depth."""
    inv = np.where((depth > 0) & (depth <= D_FAR), 1.0 / np.maximum(depth, D_NEAR), 0.0)
    return np.clip((inv - 1 / D_FAR) / (1 / D_NEAR - 1 / D_FAR), 0, 1) * (inv > 0)


def decode_depth(v):
    """Metric depth (m; NaN where empty) from an 8-bit inverse-depth channel (control_frame R, depth_frame)."""
    v = v.astype(np.float32) / 255
    return np.where(v > 0, 1 / (v * (1 / D_NEAR - 1 / D_FAR) + 1 / D_FAR), np.nan)


def control_frame(depth, sem, wh=None, shade=None):
    """Control video frame: R inverse depth, G robot, B object. G is 255 on the robot mask, or with `shade` (Lambert
    luminance of the robot mesh, sim/render.render(shade=True)) SHADE_FLOOR + (1 - SHADE_FLOOR) * luminance there."""
    r = _inverse_depth(depth)
    g = (sem == ROBOT).astype(np.float32)
    if shade is not None:  # (H, W) luminance or the (H, W, 3) shaded RGB of sim/render.render(shade=True)
        shade = shade.mean(-1) if shade.ndim == 3 else shade
        g = g * (SHADE_FLOOR + (1 - SHADE_FLOOR) * np.clip(shade, 0, 1))
    rgb = np.stack([r, g, sem == CWM_OBJ], -1).astype(np.float32)
    return cv2.resize((rgb * 255).astype(np.uint8), wh or SIZE, interpolation=cv2.INTER_NEAREST)


def depth_frame(depth, wh=None):
    """Depth proxy frame: control_frame's R channel (inverse depth) as a grey image in all three channels."""
    r = _inverse_depth(depth)
    return cv2.resize((np.repeat(r[..., None], 3, -1) * 255).astype(np.uint8), wh or SIZE,
                      interpolation=cv2.INTER_NEAREST)


def first_frame_hole(robot0, held0):
    """First-frame footprint (H, W) bool of the robot and the held objects (their masks), dilated by 7 px at 1280
    wide."""
    s = robot0.shape[1] / 1280
    k = 2 * max(1, round(7 * s)) + 1
    return cv2.dilate((robot0 | held0).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def world_rgb_frame(first, rgb, dyn, hole, wh=None, hole_late=None, late=False):
    """RGB proxy frame: a full render of the proxy world. The first frame is the static scene (the camera does not
    move); `rgb` (H, W, 3) in [0, 1], from sim/render.render(shade=True), gives the robot mesh and the objects
    (first-frame point colours) on the pixels `dyn` (robot + held objects, visible); `hole` is their first-frame
    footprint (first_frame_hole) taken from the proxy's own render at t = 0, so it is covered by what is drawn while
    nothing has moved. Footprint pixels farther than 15 px (at 1280 wide) from anything drawn now are MARK_RGB: nothing
    known is behind them, the video model has to complete them. Near the drawn parts the first frame stays, so there is
    no marker rim around things that have not moved. `hole_late`: further footprint pixels (the part of the first-frame
    segmentation the proxy's silhouette does not cover), marked only once `late` (something has moved)."""
    k = 2 * max(1, round(15 * first.shape[1] / 1280)) + 1
    near = cv2.dilate(dyn.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    out = first.copy()
    mark = hole | hole_late if (late and hole_late is not None) else hole
    out[mark & ~near] = MARK_RGB
    out[dyn] = (np.clip(rgb[dyn], 0, 1) * 255).astype(np.uint8)
    return cv2.resize(out, wh, interpolation=cv2.INTER_LINEAR) if wh else out


def ep_raw_dir(idx):
    """Raw-release directory name of DROID episode idx (index into the manifest's usable episodes)."""
    return [r for r in droid_data.manifest() if r["status"] == "ok"][idx]["raw_dir"]


def raw_frames(idx, ep, need, stereo_dir=STEREO_DIR):
    """Full-resolution ext1 frames {t: frame} for the steps in `need` from the raw MP4, or None if it is missing or not
    the loader's stream. Only the needed frames are kept (a whole 720p episode does not fit in memory). With the RLDS
    source the MP4 comes from the stereo download (`stereo_dir`)."""
    if droid_data.SOURCE == "raw":  # the loader already decodes this MP4: same stream, same length
        f = droid_data.DROID_RAW / ep_raw_dir(idx) / "ext1.mp4"
    else:
        stereo_dir = Path(stereo_dir)
        index = json.loads((stereo_dir / "index.json").read_text())
        if str(idx) not in index:
            return None
        d = stereo_dir / index[str(idx)]
        meta = json.loads(Path(glob.glob(str(d / "metadata_*.json"))[0]).read_text())
        f = d / "recordings/MP4" / Path(meta["ext1_mp4_path"]).name
    if not f.exists():
        return None
    r = imageio.get_reader(f)
    if r.count_frames() != len(ep["frames"]):
        return None
    need, out = set(need), {}
    for t, im in enumerate(r):
        if t in need:
            out[t] = im
        if len(out) == len(need):
            break
    small = cv2.resize(out[0], ep["frames"][0].shape[1::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    if np.abs(small - ep["frames"][0].astype(np.float32)).mean() > RAW_MATCH_MAX:
        return None
    return out


def raw_first_frame(idx, ep):
    """The full-resolution ext1 first frame (720x1280 RGB) of episode idx (the background of the RGB proxy, and the
    simulator's first frame with `config.cfg.hires`), or an error: never a silent fallback to the 320x180 frame. Read with cv2 from the MP4 the raw loader decodes (droid/data.py), so it is the loader's frame 0
    at full resolution; checked against ep["frames"][0] (INTER_AREA to its size). raw_frames cannot be used: it
    compares the MP4 length with len(ep["frames"]), which a clip-only run has cut. Returns (frame, report)."""
    if droid_data.SOURCE != "raw":
        raise RuntimeError(f"the full-resolution first frame needs the raw DROID source ({droid_data.SOURCE})")
    f = droid_data.DROID_RAW / ep_raw_dir(idx) / "ext1.mp4"
    if not f.exists():
        raise RuntimeError(f"full-resolution first frame: {f} missing")
    cap = cv2.VideoCapture(str(f))
    good, im = cap.read()
    cap.release()
    if not good:
        raise RuntimeError(f"full-resolution first frame: cannot decode {f}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    lo = ep["frames"][0]
    small = cv2.resize(im, lo.shape[1::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    mad = float(np.abs(small - lo.astype(np.float32)).mean())
    # a mean difference passes for frames of different moments when most of the scene is static: the worst 16x16
    # block (at the loader's size) must match as well
    d = np.abs(small - lo.astype(np.float32)).mean(-1)
    k = 16
    patch = max(float(d[y:y + k, x:x + k].mean())
                for y in range(0, d.shape[0] - k + 1, 8) for x in range(0, d.shape[1] - k + 1, 8))
    if mad > RAW_MATCH_MAX or patch > RAW_MATCH_PATCH_MAX:
        raise RuntimeError(f"full-resolution first frame: raw first frame of {idx} does not match the loader's (mean abs diff {mad:.1f}, "
                           f"worst 16x16 block {patch:.1f})")
    return im, dict(first_frame_source=str(f), first_frame_size=list(im.shape[:2]), first_frame_match_mad=mad,
                    first_frame_match_patch=patch)


def _link_pair(out, idx, src, size, n):
    """--link: symlink the raw ext1 MP4 and the RGB / depth proxies; False if one is missing or the recording is
    short."""
    W, H = size
    cond = src / "conditions"
    refs = {d: cond / f"ref_{kind}_{W}x{H}.mp4" for d, kind in (("refs", "rgb"), ("refs_depth", "depth"))}
    f = droid_data.DROID_RAW / ep_raw_dir(idx) / "ext1.mp4"
    cap = cv2.VideoCapture(str(f))
    frames, (good, first) = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.read()
    cap.release()
    if not all(r.exists() for r in refs.values()) or not good or frames < n:
        return False
    for d, r in refs.items():
        (out / d / f"{idx}.mp4").symlink_to(r.resolve())
    (out / "videos" / f"{idx}.mp4").symlink_to(f.resolve())
    imageio.imwrite(out / "first" / f"{idx}.png", cv2.resize(first[..., ::-1], (W, H), interpolation=cv2.INTER_AREA))
    return True


def _encode_pair(out, idx, src, ep, size, n, raw, stereo_dir):
    """Re-encoded proxy and real video (default / --raw), the real frames read from the raw MP4; False if the raw MP4
    or the raw-resolution proxy is missing."""
    W, H = size
    cond = src / "conditions"
    T = len(ep["frames"])
    fps = ep["fps"] if raw else CWM_FPS
    src_t = np.minimum(np.arange(n) if raw else np.round(np.arange(n) * ep["fps"] / CWM_FPS).astype(int), T - 1)
    frames = raw_frames(idx, ep, set(src_t.tolist()) | {0}, stereo_dir)  # never the loader's 320x180 frames
    if frames is None:
        print("skip (no matching raw mp4)", idx, flush=True)
        return False
    real = [cv2.resize(frames[t], (W, H), interpolation=cv2.INTER_AREA) for t in src_t]
    if raw:
        hi = cond / f"ref_{W}x{H}.mp4"  # rendered at the raw resolution by droid/run.py
        if not hi.exists():
            print("skip (no raw-resolution proxy)", idx, flush=True)
            return False
        ref = read_frames(hi, n)
    else:
        ref = [control_frame(np.fromfile(cond / f"{k:06d}.depth.f32", np.float32).reshape(CWM_HW),
                             np.array(Image.open(cond / f"{k:06d}.semantic_id.png")), (W, H)) for k in range(n)]
    write_video(out / "refs" / f"{idx}.mp4", ref, fps)
    write_video(out / "videos" / f"{idx}.mp4", real, fps)
    imageio.imwrite(out / "first" / f"{idx}.png", real[0])
    return True


def build(run, out, idxs, size=SIZE, clip_frames=CLIP_FRAMES, raw=False, link=False, stereo_dir=STEREO_DIR):
    """Write the dataset of episodes `idxs` of proxy run `run` to `out` (see the module docstring)."""
    out, n = Path(out), clip_frames
    assert not link or droid_data.SOURCE == "raw", "--link needs the raw DROID source"
    for d in ("refs", "videos", "first") + (("refs_depth",) if link else ()):
        (out / d).mkdir(parents=True, exist_ok=True)
    ok = [r for r in droid_data.manifest() if r["status"] == "ok"]
    rows = []
    for idx in idxs:
        src = Path(run) / f"droid_{idx:05d}"
        # link mode reads only the raw-resolution proxies (the 24 fps conditions/ frames may stop before n)
        if not ((src / "info.json").exists() if link else (src / "conditions" / f"{n - 1:06d}.depth.f32").exists()):
            print("skip (no conditions)", idx, flush=True)
            continue
        if (out / "videos" / f"{idx}.mp4").exists():
            text = (src / "prompt.txt").read_text() if (src / "prompt.txt").exists() else ""
        elif link:
            if not _link_pair(out, idx, src, size, n):
                print("skip (no RGBD proxy or short recording)", idx, flush=True)
                continue
            text = (src / "prompt.txt").read_text()
        else:
            ep = droid_data.load(ok[idx])
            if not _encode_pair(out, idx, src, ep, size, n, raw, stereo_dir):
                continue
            text = ep["text"]
        rows.append(dict(caption=prompt(text), video=f"videos/{idx}.mp4", reference_video=f"refs/{idx}.mp4", idx=idx))
        print("pair", idx, flush=True)
    write_jsonl(out / "dataset.jsonl", rows)
    if link:  # the same pairs with the depth proxy as the reference
        write_jsonl(out / "dataset_depth.jsonl", [dict(r, reference_video=f"refs_depth/{r['idx']}.mp4") for r in rows])
    print(len(rows), "pairs ->", out / "dataset.jsonl")


def main():
    p = parser(__doc__)
    p.add_argument("run", help="proxy run directory (droid/run.py output, droid_XXXXX/ per episode)")
    p.add_argument("out", help="dataset directory to write")
    p.add_argument("idxs", nargs="+", type=int, metavar="IDX", help="episode indices (@FILE reads them from a file)")
    p.add_argument("--size", type=size_arg, default=SIZE, metavar="WxH", help="output size (default %(default)s)")
    p.add_argument("--clip-frames", type=int, default=CLIP_FRAMES, help="clip length in frames (default %(default)s)")
    p.add_argument("--raw", action="store_true", help="full-resolution raw ext1 video at its own frame rate")
    p.add_argument("--link", action="store_true", help="link the raw video and the RGB / depth proxies, no re-encoding")
    p.add_argument("--stereo-dir", type=Path, default=STEREO_DIR,
                   help="stereo download root with the raw MP4s for --raw with the RLDS source (default %(default)s)")
    a = p.parse_args()
    build(a.run, a.out, a.idxs, a.size, a.clip_frames, a.raw, a.link, a.stereo_dir)


if __name__ == "__main__":
    main()
