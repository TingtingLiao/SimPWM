"""GR1 comparison with DreamDojo on its 100 test windows (supplementary GR1 table) and qualitative grids.

score: both models against the SAME ground-truth frames (DreamDojo's own <k>_gt.mp4) on the same field of view.
  protocol   DreamDojo's eval transform (groot_dreams VideoCrop.center_crop + VideoResize): from the 1280x800 frame,
             columns 107..1172 (int(800 * 4/3) = 1066 wide, offset (1280 - 1066) // 2), full height, torchvision
             bilinear resize with antialias to 640x480
  GT         DreamDojo's <k>_gt.mp4 (frame j = raw base + 2 + 2j), checked against that transform applied to the raw
             video (mean abs difference on frames 0 / 24 / 48 of every window)
  DreamDojo  its <k>_pred.mp4, unchanged
  SimPWM        our 800x480 generation = the full 1280x800 view scaled by (0.625, 0.6): the same region is x in
             [66.875, 733.125), full height; resampled sub-pixel exactly to 640x480 with an affine warp (bilinear).
             Alignment check: phase-correlation shift of our frame 0 against GT frame 0 (images in --figs)
  metrics    piq PSNR / SSIM / LPIPS per window over the 49 frames (as DreamDojo's action_conditioned.py), paired
             differences with 95% CIs and win rates; FVD (I3D, frames 0, 3, ..., 45 at 224x224) of each model against
             the 100 GT clips with jackknife SEs (only indicative on 100 clips); robot IoU (robot masks of the SAM 3
             instances touching the FK render mapped into the crop, generated vs GT, frames 0, 8, ..., 48,
             pixel-weighted)
grids: rows = frames 0, 16, 32, 48; columns = real | SimPWM | DreamDojo | RGB proxy (SimPWM / real / proxy: 4:3 centre crop
of 800x480; DreamDojo: its own pred); the windows with the lowest / median / highest SimPWM PSNR unless ids are given.

Inputs: --gen = scripts/gr1/generate.sbatch output (shard_*/order.json + samples/step_000000_<k>.mp4, panels
[reference | generation]), --dreamdojo = scripts/gr1/dreamdojo.sbatch output (shard_*/iter_000050000/<k>_{gt,pred}
.mp4),
--proxies = `simpwm.gr1.run eval` output (w<k>/robot.npz, video.mp4, ref.mp4). Needs piq (0.8.0, DreamDojo's version).
usage: python -m simpwm.eval.gr1_bench score [--gen DIR] [--dreamdojo DIR] [--proxies DIR] [--out JSON] [--no-robot]
       python -m simpwm.eval.gr1_bench grids [--gen DIR] [--dreamdojo DIR] [--proxies DIR] [--score JSON] [--out DIR]
                                          [K ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
import torch

from simpwm.gr1 import calib as calib_gr1
from simpwm.gr1 import data as gr1_data
from simpwm.paths import RUNS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DD_ITER = "iter_000050000"  # DreamDojo's output subdir (official GR1 post-train checkpoint, iteration 50000)
X0, W0 = 107, 1066  # DreamDojo's crop of the 1280x800 frame
S = 0.625  # our 800x480 output: x scale of the full view
FVD_FRAMES = range(0, 48, 3)
ROBOT_EVERY = 8
RUN = RUNS / "gr1"
_L = {}


def read(p, n=49):
    r = iio.get_reader(str(p), "ffmpeg")
    fr = [f for _, f in zip(range(n), r)]
    r.close()
    return np.stack(fr)


def simpwm_path(gen_dir, k):
    """Generated sample of window k (the shard whose order.json lists it)."""
    for sh in Path(gen_dir).glob("shard_*"):
        o = json.loads((sh / "order.json").read_text())
        if k in o:
            return sh / "samples" / f"step_000000_{o.index(k) + 1}.mp4"


def dd_dir(dd_root, k):
    """DreamDojo's output dir holding window k (the shard that ran it: the others hold placeholders)."""
    return next(Path(dd_root).glob(f"shard_*/{DD_ITER}/{k:04d}_gt.mp4")).parent


def robot_masks(proxies, k):
    """(49, 500, 800) FK render masks of window k (gr1/run.py robot.npz)."""
    z = np.load(Path(proxies) / f"w{k:03d}" / "robot.npz")
    return np.unpackbits(z["m"], axis=-1)[..., :z["shape"][-1]].astype(bool)


# ------------------------------------------------------------------ score
def dd_transform(frames):
    """DreamDojo's eval transform of raw 1280x800 frames (T, H, W, 3) uint8 -> (T, 480, 640, 3) uint8."""
    import torchvision.transforms as T

    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255
    x = x[:, :, :, X0:X0 + W0]
    x = T.Resize((480, 640), interpolation=T.InterpolationMode.BILINEAR, antialias=True)(x)
    return (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()


def ours_to_protocol(frames):
    """(T, 480, 800, 3) -> (T, 480, 640, 3): source x = X0*S + (u + 0.5) * W0*S/640 - 0.5, y = v."""
    sx = W0 * S / 640
    M = np.float32([[sx, 0, X0 * S + 0.5 * sx - 0.5], [0, 1, 0]])
    return np.stack([cv2.warpAffine(f, M, (640, 480), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_REPLICATE) for f in frames])


def map_mask(m):
    """FK render mask (500x800, full view) -> protocol 640x480."""
    sx = W0 * S / 640
    M = np.float32([[sx, 0, X0 * S + 0.5 * sx - 0.5], [0, 500 / 480, 0.5 * 500 / 480 - 0.5]])
    return cv2.warpAffine(m.astype(np.uint8), M, (640, 480), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0


def metrics(pred, gt):
    import piq

    n = min(len(pred), len(gt))
    x = torch.from_numpy(pred[:n]).float().div(255).permute(0, 3, 1, 2).to(DEV)
    y = torch.from_numpy(gt[:n]).float().div(255).permute(0, 3, 1, 2).to(DEV)
    if "lp" not in _L:
        _L["lp"] = piq.LPIPS().to(DEV)
    with torch.no_grad():
        # LPIPS in batches of 8 frames, frame-weighted = DreamDojo's one-batch mean over the 49 frames
        lp = sum(_L["lp"](x[i:i + 8], y[i:i + 8]).mean().item() * len(x[i:i + 8]) for i in range(0, n, 8)) / n
        return dict(psnr=piq.psnr(x, y).mean().item(), ssim=piq.ssim(x, y).mean().item(), lpips=lp)


def i3d(frames):
    from simpwm.eval import wm_bench

    assert len(wm_bench.FVD_STEPS) == len(FVD_FRAMES)
    return wm_bench.i3d_features(dict(zip(wm_bench.FVD_STEPS, (frames[t] for t in FVD_FRAMES))))


def robot_mask(img, rend):
    return calib_gr1.robot_mask(calib_gr1.robot_instances(img), rend)


def ci(d):
    d = np.asarray(d, float)
    return dict(mean=float(d.mean()), ci95=float(1.96 * d.std(ddof=1) / np.sqrt(len(d))))


def align_figure(gt, ours, dd):
    tiles = [gt[0], ours[0], (0.5 * gt[0] + 0.5 * ours[0]).astype(np.uint8),
             np.clip(3 * np.abs(gt[0].astype(int) - ours[0].astype(int)), 0, 255).astype(np.uint8),
             np.clip(3 * np.abs(gt[0].astype(int) - dd[0].astype(int)), 0, 255).astype(np.uint8)]
    img = np.concatenate(tiles, 1)
    for i, s in enumerate(("GT f0", "ours f0 (unified crop)", "blend", "|GT-ours| x3", "|GT-DD| x3")):
        cv2.putText(img, s, (640 * i + 8, 28), 0, 0.8, (255, 255, 0), 2)
    return img


def score(a):
    from simpwm.eval.wm_bench import frechet

    a.figs.mkdir(parents=True, exist_ok=True)
    windows = {e["k"]: e for e in json.loads(a.windows.read_text())}
    rows, F, ver = {}, {"simpwm": [], "dd": [], "gt": []}, []
    for k in range(len(windows)):
        d = dd_dir(a.dreamdojo, k)
        gt = read(d / f"{k:04d}_gt.mp4")
        dd = read(d / f"{k:04d}_pred.mp4")
        ours = ours_to_protocol(read(simpwm_path(a.gen, k))[:, :, -800:])
        # GT check: DreamDojo's transform of the raw frames base + 2 + 2j
        e = windows[k]
        js = [0, 24, 48]
        raw = gr1_data.read_frames(e["ep"], [e["base"] + 2 + 2 * j for j in js])
        ref = dd_transform(np.stack([raw[e["base"] + 2 + 2 * j] for j in js]))
        ver.append(float(np.abs(ref.astype(float) - gt[js].astype(float)).mean()))
        # frame-0 alignment of our frame 0 to GT frame 0
        g0 = cv2.cvtColor(gt[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
        o0 = cv2.cvtColor(ours[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
        win = cv2.createHanningWindow(g0.shape[::-1], cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(g0, o0, win)
        r = dict(simpwm=metrics(ours, gt), dreamdojo=metrics(dd, gt), gt_check_mae=ver[-1], f0_shift=[dx, dy],
                 f0_mae_simpwm=float(np.abs(ours[0].astype(float) - gt[0].astype(float)).mean()),
                 f0_mae_dd=float(np.abs(dd[0].astype(float) - gt[0].astype(float)).mean()))
        F["simpwm"].append(i3d(ours)), F["dd"].append(i3d(dd)), F["gt"].append(i3d(gt))
        if not a.no_robot:
            m = robot_masks(a.proxies, k)
            acc = {"simpwm": [0.0, 0.0], "dreamdojo": [0.0, 0.0]}
            for t in range(0, 49, ROBOT_EVERY):
                rend = map_mask(m[t])
                mg = robot_mask(gt[t], rend)
                for name, v in (("simpwm", ours), ("dreamdojo", dd)):
                    mv = robot_mask(v[t], rend)
                    acc[name][0] += float((mv & mg).sum())
                    acc[name][1] += float((mv | mg).sum())
            r["robot"] = acc
        if k in (0, 50, 77):
            cv2.imwrite(str(a.figs / f"align_w{k:03d}.jpg"), align_figure(gt, ours, dd)[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows[k] = r
        print(k, {m_: round(r["simpwm"][m_] - r["dreamdojo"][m_], 3) for m_ in ("psnr", "ssim", "lpips")},
              "gt_check", round(ver[-1], 2), "shift", np.round([dx, dy], 2).tolist(), flush=True)
    ks = sorted(rows)
    summ = {}
    for m_ in ("psnr", "ssim", "lpips"):
        x = np.array([rows[k]["simpwm"][m_] for k in ks])
        y = np.array([rows[k]["dreamdojo"][m_] for k in ks])
        better = (x > y) if m_ != "lpips" else (x < y)
        summ[m_] = dict(simpwm=ci(x), dreamdojo=ci(y), diff_simpwm_minus_dd=ci(x - y), simpwm_win_rate=float(better.mean()))
    fw, fd, fg = (np.stack(F[x]) for x in ("simpwm", "dd", "gt"))
    n = len(ks)

    def jack(f):
        v = np.array([frechet(np.delete(f, i, 0), np.delete(fg, i, 0)) for i in range(n)])
        return float(np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum()))

    summ["fvd"] = dict(simpwm=frechet(fw, fg), dreamdojo=frechet(fd, fg), simpwm_jackknife_se=jack(fw),
                       dd_jackknife_se=jack(fd))
    if not a.no_robot:
        rw = np.array([rows[k]["robot"]["simpwm"] for k in ks])
        rd = np.array([rows[k]["robot"]["dreamdojo"] for k in ks])
        pw, pd = rw[:, 0] / np.maximum(rw[:, 1], 1), rd[:, 0] / np.maximum(rd[:, 1], 1)
        summ["robot_iou"] = dict(simpwm=float(rw[:, 0].sum() / rw[:, 1].sum()),
                                 dreamdojo=float(rd[:, 0].sum() / rd[:, 1].sum()),
                                 per_window_diff=ci(pw - pd), simpwm_win_rate=float((pw > pd).mean()))
    summ["checks"] = dict(gt_vs_raw_transform_mae=ci(ver), gt_vs_raw_max=float(max(ver)),
                          f0_shift_px_mean_abs=np.abs([rows[k]["f0_shift"] for k in ks]).mean(0).tolist(),
                          f0_mae_simpwm=ci([rows[k]["f0_mae_simpwm"] for k in ks]),
                          f0_mae_dd=ci([rows[k]["f0_mae_dd"] for k in ks]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(protocol=__doc__, summary=summ, windows=rows), indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


# ------------------------------------------------------------------ grids
def crop(f):
    """4:3 centre crop (640x480) of a frame resized to 800x480."""
    f = cv2.resize(f, (800, 480), interpolation=cv2.INTER_AREA) if f.shape[:2] != (480, 800) else f
    return f[:, 80:720]


def grids(a):
    a.out.mkdir(parents=True, exist_ok=True)
    res = json.loads(a.score.read_text())["windows"]
    ks = a.windows_ids
    if not ks:
        ps = sorted((v["simpwm"]["psnr"], int(k)) for k, v in res.items())
        ks = [ps[0][1], ps[len(ps) // 2][1], ps[-1][1]]
    windows = {e["k"]: e for e in json.loads(a.windows.read_text())}
    for k in ks:
        real = read(a.proxies / f"w{k:03d}" / "video.mp4")
        prox = read(a.proxies / f"w{k:03d}" / "ref.mp4")
        w = [f[:, -800:] for f in read(simpwm_path(a.gen, k))]
        d = read(dd_dir(a.dreamdojo, k) / f"{k:04d}_pred.mp4")
        g = np.concatenate([np.concatenate([crop(real[t]), crop(w[t]), d[t], crop(prox[t])], 1)
                            for t in (0, 16, 32, 48)], 0).copy()
        for i, s in enumerate(("real", "SimPWM-GR1 (ours)", "DreamDojo", "RGB proxy")):
            cv2.putText(g, s, (640 * i + 10, 30), 0, 0.9, (255, 255, 0), 2)
        e = windows[k]
        rw, rd = res.get(str(k), {}).get("simpwm", {}), res.get(str(k), {}).get("dreamdojo", {})
        cv2.putText(g, f"window {k}: ep {e['ep']} raw {e['start']}..{e['start'] + 96} step 2 | PSNR ours "
                       f"{rw.get('psnr', 0):.2f} DD {rd.get('psnr', 0):.2f}", (10, g.shape[0] - 12), 0, 0.8,
                    (255, 255, 0), 2)
        cv2.imwrite(str(a.out / f"grid_w{k:03d}.jpg"), g[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(a.out / f"grid_w{k:03d}.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("score", "PSNR / SSIM / LPIPS / FVD / robot IoU of SimPWM and DreamDojo"),
                      ("grids", "qualitative comparison grids")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--gen", type=Path, default=RUN / "gen", help="SimPWM generations (scripts/gr1/generate.sbatch)")
        p.add_argument("--dreamdojo", type=Path, default=RUN / "dreamdojo",
                       help="DreamDojo outputs (scripts/gr1/dreamdojo.sbatch)")
        p.add_argument("--proxies", type=Path, default=RUN / "proxies_eval", help="`simpwm.gr1.run eval` output")
        p.add_argument("--windows", type=Path, default=gr1_data.TEST_WINDOWS, help="test windows json")
    p = sub.choices["score"]
    p.add_argument("--out", type=Path, default=RUN / "bench.json", help="output json")
    p.add_argument("--figs", type=Path, default=RUN / "bench", help="frame-0 alignment images")
    p.add_argument("--no-robot", action="store_true", help="skip the robot IoU (no SAM 3)")
    p = sub.choices["grids"]
    p.add_argument("--score", type=Path, default=RUN / "bench.json", help="`score` output (picks the windows)")
    p.add_argument("--out", type=Path, default=RUN / "grids", help="output dir")
    p.add_argument("windows_ids", type=int, nargs="*", metavar="K", help="window ids (default: min / median / max "
                                                                          "SimPWM PSNR)")
    a = ap.parse_args()
    if a.cmd == "score":
        score(a)
    else:
        grids(a)


if __name__ == "__main__":
    main()
