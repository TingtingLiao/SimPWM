"""GR1 proxies: the RGB proxy (LTX reference video) and the real clip of training items and test windows.

Item = (episode, raw start frame, N frames at stride 2 = 10 fps). Per item:
  camera     per-frame head camera = FK(waist + neck)(t) @ the session's head->camera offset (gr1/data.camera_calib)
  robot      GR1T1 + Fourier hands, GPU-rasterized and Lambert-shaded at 500x800; the hand joints are scaled per hand
             by --hand-scale (the Fourier-hand mapping under-closes the right hand; fitted with gr1/hand_scale.py)
  scene      the first frame lifted to 3D with Depth Anything V2 metric depth (scaled with the FK robot as the ruler:
             median render z / mono z on the robot's first-frame footprint) and re-splatted (z-buffered 2x2 points)
             into every later camera; a static camera (waist still) keeps the first frame as it is
  proxy RGB  warped first frame + robot; magenta where nothing is known (no scene point, or the robot's first-frame
             footprint farther than 15 px @1280 from the robot drawn now): video/dataset.world_rgb_frame's convention
  arm IoU    robot mask of the SAM 3 instances touching the render vs the render (gr1/calib.py), every 8th clip frame,
             pixel-weighted
  refit      training items only: the camera is refitted on the clip's own IoU frames (the 9 parameters of
             calib.apply, Nelder-Mead at 250x400) and kept if it improves the IoU; test windows keep the session camera
Writes OUT/<name>/: video.mp4 (real clip, 1280x768, 10 fps), ref.mp4 (proxy, 800x480), first.png (1280x768),
prompt.txt, info.json, and robot.npz (packed 500x800 render masks, test windows only). Training items below --min-iou
get only info.json (skipped=true).

usage: python -m simpwm.gr1.run train OUT [--items CANDIDATES_JSON] [--shard K --num-shards N] [--hand-scale L:R]
       python -m simpwm.gr1.run eval OUT [--items TEST_WINDOWS_JSON] ...
  train: items [{ep, start}] from gr1/select.py, name = <ep>, 97 frames
  eval:  items [{k, ep, base, start}] (DreamDojo's test windows), name = w<k>, 49 frames (raw base + 2 + 2j)
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from simpwm.core.raster_gpu import GpuRaster
from simpwm.gr1 import calib as calib_gr1
from simpwm.gr1 import data as gr1_data
from simpwm.gr1.embodiment import SL, GR1Embodiment
from simpwm.gr1.session_calib import Scorer, camera_objective, nelder_mead
from simpwm.paths import RUNS

IOU_EVERY = 8  # arm IoU on every 8th clip frame
MARK = np.array([255, 0, 255], np.uint8)
HW = gr1_data.WORK_HW


def scale_hands(q, fs):
    """Hand joints of states q scaled by fs = (left, right)."""
    q = np.array(q, float, copy=True)
    q[..., SL["left_hand"]] *= fs[0]
    q[..., SL["right_hand"]] *= fs[1]
    return q


def splat(P, cam):
    """z-buffered 2x2 px splat of world points P (n, 3) into cam -> (H, W) index of the nearest point, -1 = none."""
    H, W = cam.H, cam.W
    uv, z = cam.project(P)
    ok = np.nonzero((z > 0.05) & np.isfinite(uv).all(1))[0]
    zb = np.full(H * W, np.inf)
    win = np.full(H * W, -1, np.int64)
    for du in (0, 1):
        for dv in (0, 1):
            u = np.floor(uv[ok, 0] - 0.5 * du).astype(int) + du
            v = np.floor(uv[ok, 1] - 0.5 * dv).astype(int) + dv
            m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            idx, zz, pid = v[m] * W + u[m], z[ok][m], ok[m]
            order = np.argsort(-zz)  # far first: among duplicates the nearest is written last and wins
            idx, zz, pid = idx[order], zz[order], pid[order]
            closer = zz < zb[idx]
            zb[idx[closer]] = zz[closer]
            win[idx[closer]] = pid[closer]
    return win.reshape(H, W)


def arm_iou(inst, rmask):
    """Pixel-weighted arm IoU over the frames of `inst` ({k: SAM 3 instances}) against the render masks rmask[k]."""
    per = {}
    for k, ins in inst.items():
        r = rmask[k]
        m = calib_gr1.robot_mask(ins, r)
        if m.any() or r.any():
            per[int(k)] = (int((m & r).sum()), int((m | r).sum()))
    if not per:
        return dict(iou=None, iou_frames=None, frames={})
    i, u = np.array(list(per.values()), float).T
    return dict(iou=float(i.sum() / max(u.sum(), 1)), iou_frames=float(np.mean(i / np.maximum(u, 1))),
                frames={str(k): v for k, v in per.items()})


class Worker:
    def __init__(self, cam_dir=gr1_data.SESSION_CAMERAS, hand_scale=(1.0, 1.0), min_iou=0.6, episode_refit=True):
        self.emb = GR1Embodiment(verbose=False)
        self.gr = GpuRaster(self.emb)
        self.cam_dir, self.fs = cam_dir, np.asarray(hand_scale, float)
        self.min_iou, self.episode_refit = min_iou, episode_refit

    def refit(self, calib, q, inst):
        """Camera refit on the IoU frames of one clip (gr1/session_calib.Scorer at 250x400)."""
        sc = Scorer(self.emb, self.gr, [(q[k], v) for k, v in inst.items()], (HW[0] // 2, HW[1] // 2))
        step = np.r_[0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.03, 0.03, 0.02]
        x = nelder_mead(camera_objective(calib, sc), np.zeros(9), step, 300)
        return dict(calib_gr1.apply(calib, x), dt=calib.get("dt", 0)), x

    def robot(self, q, cams):
        rz, rm, rgb = [], [], []
        for qt, c in zip(q, cams):
            self.emb.set_state(qt)
            z, part, col = self.gr.render(c, shade=True)
            rz.append(z), rm.append(part >= 0), rgb.append(col)
        return rz, rm, rgb

    def proxy(self, frames, cams, rz, rm, rgb):
        from simpwm import perception

        mono = perception.metric_depth(frames[0])
        fp = rm[0] & (rz[0] > 0)
        scale = float(np.median(rz[0][fp] / np.maximum(mono[fp], 1e-3))) if fp.sum() > 200 else 0.36
        P = cams[0].unproject(mono * scale).reshape(-1, 3)
        col0 = frames[0].reshape(-1, 3)
        k = 2 * max(1, round(7 * HW[1] / 1280)) + 1
        foot0 = cv2.dilate(rm[0].astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        static = all(np.allclose(c.T_wc, cams[0].T_wc, atol=3e-3) for c in cams)
        kn = 2 * max(1, round(15 * HW[1] / 1280)) + 1
        out = []
        for i in range(len(cams)):
            if static:
                bg, hit, foot = frames[0].copy(), np.ones(HW, bool), foot0
            else:
                win = splat(P, cams[i])
                hit = win >= 0
                bg = col0[np.maximum(win, 0)].reshape(HW + (3,))
                foot = hit & foot0.reshape(-1)[np.maximum(win, 0)]
            near = cv2.dilate(rm[i].astype(np.uint8), np.ones((kn, kn), np.uint8)).astype(bool)
            o = bg.copy()
            o[~hit] = MARK
            o[foot & ~near] = MARK
            o[rm[i]] = (np.clip(rgb[i][rm[i]], 0, 1) * 255).astype(np.uint8)
            out.append(o)
        return out, dict(depth_scale=scale, static_camera=bool(static))

    def item(self, mode, out_root, it):
        t0 = time.time()
        ep, start = int(it["ep"]), int(it["start"])
        n = gr1_data.CLIP_FRAMES[mode]
        name = str(ep) if mode == "train" else f"w{it['k']:03d}"
        out = Path(out_root) / name
        if (out / "info.json").exists():
            return None
        out.mkdir(parents=True, exist_ok=True)
        q_all = gr1_data.states(ep)
        ts = np.minimum(gr1_data.clip_ts(start, n), len(q_all) - 1)
        raw = gr1_data.read_frames(ep, ts)
        ts = np.minimum(ts, max(raw))  # the video may hold a frame fewer than the states
        full = [raw[t] for t in ts]
        frames = [cv2.resize(f, HW[::-1], interpolation=cv2.INTER_AREA) for f in full]
        calib, src = gr1_data.camera_calib(ep, self.cam_dir)
        q = scale_hands(q_all[np.clip(ts + int(calib.get("dt", 0)), 0, len(q_all) - 1)], self.fs)
        cams = gr1_data.cameras(self.emb, q, calib)
        rz, rm, rgb = self.robot(q, cams)
        inst = {k: calib_gr1.robot_instances(frames[k]) for k in range(0, n, IOU_EVERY)}
        iou = iou0 = arm_iou(inst, rm)
        x = None
        if mode == "train" and self.episode_refit and iou0["iou"] is not None:
            c1, x = self.refit(calib, q, inst)
            cams1 = gr1_data.cameras(self.emb, q, c1)
            rz1, rm1, rgb1 = self.robot(q, cams1)
            iou1 = arm_iou(inst, rm1)
            if iou1["iou"] is not None and iou1["iou"] > iou0["iou"]:
                calib, cams, rz, rm, rgb, iou, src = c1, cams1, rz1, rm1, rgb1, iou1, src + "+episode_refit"
        info = dict(ep=ep, start=start, n=n, stride=gr1_data.STRIDE, ts=ts.tolist(), session=gr1_data.session(ep),
                    unit=gr1_data.unit(ep), calib_source=src, arm_iou=iou["iou"], arm_iou_frames=iou["iou_frames"],
                    iou_detail=iou["frames"], arm_iou_session=iou0["iou"], refit_x=None if x is None else x.tolist(),
                    hand_scale=self.fs.tolist(),
                    calib={k: calib[k] for k in calib_gr1.CALIB_KEYS + ("dt",) if k in calib},
                    caption=gr1_data.caption(ep), task=gr1_data.episodes()[ep]["tasks"][0])
        if mode == "eval":
            info.update(k=it["k"], base=it["base"])
        if mode == "train" and (iou["iou"] is None or iou["iou"] < self.min_iou):
            info.update(skipped=True, seconds=time.time() - t0)
            (out / "info.json").write_text(json.dumps(info, indent=1))
            return info
        px, pinfo = self.proxy(frames, cams, rz, rm, rgb)
        info.update(pinfo)
        fps = gr1_data.FPS_SRC / gr1_data.STRIDE
        imageio.mimwrite(out / "ref.mp4", [cv2.resize(p, gr1_data.REF_WH, interpolation=cv2.INTER_AREA) for p in px],
                         fps=fps, quality=9, macro_block_size=1)
        vid = [cv2.resize(f, gr1_data.SQ_WH, interpolation=cv2.INTER_AREA) for f in full]
        imageio.mimwrite(out / "video.mp4", vid, fps=fps, quality=9, macro_block_size=1)
        imageio.imwrite(out / "first.png", vid[0])
        (out / "prompt.txt").write_text(gr1_data.caption(ep))
        if mode == "eval":
            m = np.stack(rm)
            np.savez_compressed(out / "robot.npz", m=np.packbits(m, axis=-1), shape=np.array(m.shape))
        info.update(skipped=False, seconds=time.time() - t0)
        (out / "info.json").write_text(json.dumps(info, indent=1, default=float))
        return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=sorted(gr1_data.CLIP_FRAMES), help="train: training candidates; eval: test windows")
    ap.add_argument("out", type=Path, help="output dir (one subdir per item; items already done are skipped)")
    ap.add_argument("--items", type=Path, help="items json (default: train RUNS/gr1/candidates.json from "
                                               "gr1/select.py; eval resources/gr1/test_windows.json)")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard of the items")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--cameras", type=Path, default=gr1_data.SESSION_CAMERAS,
                    help="per-session camera dir (gr1/session_calib.py output)")
    ap.add_argument("--hand-scale", default="1:1", help="L:R multipliers of the hand joints (paper: 0.89:1.28)")
    ap.add_argument("--min-iou", type=float, default=0.6, help="training items below this arm IoU are skipped")
    ap.add_argument("--no-episode-refit", action="store_true", help="training items keep the session camera")
    a = ap.parse_args()
    items_json = a.items or (RUNS / "gr1/candidates.json" if a.mode == "train" else gr1_data.TEST_WINDOWS)
    items = json.loads(items_json.read_text())[a.shard::a.num_shards]
    fs = [float(x) for x in a.hand_scale.replace(",", ":").split(":")]
    w = Worker(a.cameras, fs, a.min_iou, not a.no_episode_refit)
    for it in items:
        try:
            r = w.item(a.mode, a.out, it)
            if r is not None:
                print(json.dumps({k: r[k] for k in ("ep", "start", "arm_iou", "calib_source", "skipped", "seconds")}),
                      flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print("FAILED", it, e, flush=True)


if __name__ == "__main__":
    main()
