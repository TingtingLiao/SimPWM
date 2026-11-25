"""Table 2 (the real-to-sim proxy ablation, 97-step protocol): the metrics beyond proxy_metrics.py, for the rule-based
rows (--row NAME=RUN; "full" is the full system) and the PointWorld row "pw" (evaluation only).

  VLM grasp   grasp_vlm.py's judge on each row's RGB proxy video (RUN/droid_{i5}/conditions/ref_rgb_1280x704.mp4; pw:
              --pw-videos), GT = the real-video answers of Table 3; answer sets OUT/vlm/<row>. Grasp success =
              agreement; Halluc. = the proxy names an object, the GT none.
  PointWorld  the RGB proxies rendered from PointWorld's rollouts (--pw-rollouts <idx>.npz: first-frame pixels `pix`,
              per-step point positions `xyz`, camera K / world2cam): moved points splatted over the first frame, the
              full system's robot drawn on top, magenta where a moved point left and nothing landed. RGB L1 / L2 and
              depth L2 / L1 as proxy_metrics.py except for the two inputs PointWorld has no file for: the proxy moving
              region (the full system's robot united with where moved points land) and the proxy depth (pw_render's
              z-buffer over the first-frame stereo depth, NaN in holes; in float, and through the 8-bit inverse-depth
              encoding of the rule-based depth videos: depth_l2_q8).
  object IoU  image-based, the same for every row: at steps 8, 16, ..., 96 the real grasp target = real_dyn's moving
              region minus all SAM 3 robot instances of that step (dilated 5 px), 640x352; the proxy's moving-object
              region = pixels whose RGB differs from the proxy's step 0 by more than DIFF (max over channels) that are
              neither magenta (dilated 5x5) nor the proxy's robot (control video ref_1280x704.mp4 G >= 40 of the row's
              run, pw: the full system's; dilated 7x7), opened 5x5 (a held object hidden behind the rendered gripper
              does not count). Per episode the mean IoU over the steps whose real target has >= 50 px; episodes whose
              real video has a grasp target only.
  aligned     ALIGNED_DOC: every row of an episode scored on the same pixels.
  robot IoU   pw uses the full system's robot render: its arm IoU is the full row's.

Stages:
  pw-metrics       PointWorld RGB / depth metrics -> OUT/pw_metrics/<idx>.json
  object-iou ROW   image object IoU of one row -> OUT/object_iou/<row>/<idx>.json
  aligned          aligned RGB L1 / depth L2 of every row -> OUT/aligned/<idx>.json
  summary          VLM grasp, object IoU, PointWorld and aligned metrics with proxy_metrics.py's summary (--auto)
                   -> OUT/summary.json
usage: python -m simpwm.eval.grasp_vlm_ablation STAGE [ROW] OUT --ids IDS_TXT --real-dyn DIR --row full=RUN [--row NAME=RUN
           ...] [--pw-rollouts DIR --pw-videos DIR] [--shard T N] [--auto PROXY_METRICS_SUMMARY]
"""

import json
import tempfile
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from simpwm.core.state import C_OBJ0, is_robot
from simpwm.eval import grasp_vlm as gv
from simpwm.eval import real_dyn
from simpwm.eval.common import (
    STEREO_DIR,
    STEREO_SEQ_DIR,
    ci,
    each,
    episode_dir,
    load_dir,
    magenta,
    parser,
    raw_dir,
    read_frames,
    read_ids,
    read_indices,
    real_frames,
)
from simpwm.eval.grasp_iou import held_within
from simpwm.eval.proxy_metrics import (
    CHANGE,
    DEPTH,
    DEPTH_STEPS,
    DWH,
    RGB,
    WH,
    change_mask,
    decoded_depth,
    rgb_errors,
    stereo_depth,
    valid_depth,
)
from simpwm.video.common import write_video
from simpwm.video.dataset import decode_depth, depth_frame

N = 97  # clip length (recorded steps)
EVERY = max(8, N // 15)  # real_dyn's instance steps
DIFF = 40
STEPS_OBJ = list(range(8, 97, 8))
CONTROL = "conditions/ref_1280x704.mp4"
ALIGNED_DOC = """Aligned RGB L1 / depth L2: every row of an episode scored on the same pixels, all from decoded videos.
RGB (640x352, steps 1..96): main mask U = the real moving region (real_dyn) | every rule-based row's robot and held
objects (its proxy_cat, held_within) | PointWorld's moved-point destinations, minus the union of all rows' magenta;
sensitivity mask R = the real moving region minus that magenta (proxy-independent). Depth (1280x704, steps 24/48/72/96):
every row's depth video decoded to metres (rule-based: conditions/ref_depth_1280x704.mp4; PointWorld: its z-buffer
written with video/dataset.depth_frame + imageio mp4 quality 9, as droid/run.py writes the depth proxy, then decoded);
main pixels: real stereo and all rows valid in 0.2-3 m, and moved > 2 cm from step 0 in the real or in any row;
sensitivity: same validity, moved in the real only. Per row the mean over steps, then over episodes."""


class Table2:
    """The inputs of the stages."""

    def __init__(self, out, real, rows, pw_rollouts=None, pw_videos=None, stereo_dir=STEREO_DIR,
                 stereo_seq_dir=STEREO_SEQ_DIR):
        self.out, self.real, self.rows = Path(out), Path(real), {k: Path(v) for k, v in rows.items()}
        self.pw_rollouts, self.pw_videos = pw_rollouts and Path(pw_rollouts), pw_videos and Path(pw_videos)
        self.stereo_dir, self.stereo_seq_dir = Path(stereo_dir), Path(stereo_seq_dir)
        self.names = list(self.rows) + (["pw"] if self.pw_rollouts else [])

    def run_of(self, row):
        """Proxy run of a row (pw: the full system's, whose robot it uses)."""
        return self.rows["full" if row == "pw" else row]

    def rgb_video(self, row, idx):
        return self.pw_videos / f"{idx}.mp4" if row == "pw" else episode_dir(self.rows[row], idx) / RGB


# ----------------------------------------------------------------------------------------------- PointWorld render
def pw_render(t2, idx, n=N):
    """PointWorld's RGB proxy frames, its z-buffer depth (m, NaN unknown) and its moved-point destination masks."""
    robot_g, still, H, W = 40, 0.015, 704, 1280  # PointWorld renderer constants
    z = np.load(t2.pw_rollouts / f"{idx}.npz")
    pix, xyz, K0, w2c = z["pix"].astype(int), z["xyz"].astype(np.float32), z["K"], z["world2cam"]
    disp_s = xyz - xyz[:1]
    lut = np.full((180, 320), -1, np.int64)
    lut[pix[:, 1], pix[:, 0]] = np.arange(len(pix))
    from scipy.ndimage import distance_transform_edt

    _, (iy, ix) = distance_transform_edt(lut < 0, return_indices=True)
    near = lut[iy, ix]
    far = distance_transform_edt(lut < 0) > 2
    gy, gx = np.mgrid[0:H, 0:W]
    ly, lx = np.minimum(gy * 180 // H, 179), np.minimum(gx * 320 // W, 319)
    nid = np.where(far[ly, lx], -1, near[ly, lx])
    K = K0 * np.array([[W / 320], [H / 180], [1]])
    d0 = cv2.resize(np.load(t2.stereo_dir / f"{idx}_ext1.npy").astype(np.float32), (W, H),
                    interpolation=cv2.INTER_NEAREST)
    r = imageio.get_reader(raw_dir(idx) / "ext1.mp4")
    first = cv2.resize(r.get_data(0), (W, H), interpolation=cv2.INTER_AREA)
    r.close()
    ok = (nid >= 0) & np.isfinite(d0) & (d0 > 0)
    v, u = np.nonzero(ok)
    zc = d0[v, u]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * zc, (v - K[1, 2]) / K[1, 1] * zc, zc], -1)
    c2w = np.linalg.inv(w2c)
    p0 = pc @ c2w[:3, :3].T + c2w[:3, 3]
    col = first[v, u]
    nids = nid[v, u]
    dbase = np.where(np.isfinite(d0) & (d0 > 0), d0, np.nan).astype(np.float32)
    pd = episode_dir(t2.rows["full"], idx)
    ctl, prgb, pdep = (imageio.get_reader(pd / f) for f in (CONTROL, RGB, DEPTH))
    frames, depths, dests = [], [], []
    hole0 = None
    for t, (c, pr, dp) in enumerate(zip(ctl, prgb, pdep)):
        if t >= n:
            break
        robot = c[..., 1] >= robot_g
        if hole0 is None:
            hole0 = cv2.dilate(robot.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
        s = t // 2  # rollout steps at half the video rate: odd frames interpolate
        dsp = disp_s[min(s, len(disp_s) - 1)] if t % 2 == 0 else \
            0.5 * (disp_s[min(s, len(disp_s) - 1)] + disp_s[min(s + 1, len(disp_s) - 1)])
        dd = dsp[nids]
        mv = np.linalg.norm(dd, axis=1) > still
        img, dep, dst = first.copy(), dbase.copy(), np.zeros((H, W), bool)
        if mv.any():
            img[v[mv], u[mv]] = (255, 0, 255)
            dep[v[mv], u[mv]] = np.nan
            q = (p0[mv] + dd[mv]) @ w2c[:3, :3].T + w2c[:3, 3]
            front = q[:, 2] > 0.05
            uv = q[front] @ K.T
            uu = np.round(uv[:, 0] / uv[:, 2]).astype(int)
            vv = np.round(uv[:, 1] / uv[:, 2]).astype(int)
            zz, cc = q[front, 2], col[mv][front]
            inb = (uu >= 0) & (uu < W - 1) & (vv >= 0) & (vv < H - 1)
            uu, vv, zz, cc = uu[inb], vv[inb], zz[inb], cc[inb]
            order = np.argsort(-zz)  # 2x2 splats, near overwrites far
            for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
                img[vv[order] + dv, uu[order] + du] = cc[order]
                dep[vv[order] + dv, uu[order] + du] = zz[order]
                dst[vv[order] + dv, uu[order] + du] = True
        img[hole0 & ~robot] = (255, 0, 255)
        dep[hole0 & ~robot] = np.nan
        dst[hole0 | robot] = False
        img[robot] = pr[robot]
        dep[robot] = decode_depth(dp[..., 0])[robot]
        frames.append(img)
        depths.append(dep)
        dests.append(dst)
    for x in (ctl, prgb, pdep):
        x.close()
    return frames, depths, dests


def _q8(x):
    """A float depth map through the depth proxy's 8-bit inverse-depth encoding (no video codec)."""
    return decode_depth(depth_frame(np.nan_to_num(x, nan=0.0), DWH)[..., 0])


def _video_depth(depths):
    """PointWorld's z-buffer written and decoded exactly as the rule-based depth videos (depth_frame -> mp4 q9)."""
    with tempfile.TemporaryDirectory() as td:
        fn = Path(td) / "d.mp4"
        write_video(fn, [depth_frame(np.nan_to_num(x, nan=0.0), DWH) for x in depths], 15)
        return decoded_depth(fn, DEPTH_STEPS)


def pw_metrics(t2, idx):
    """proxy_metrics.episode's RGB / depth part for the PointWorld proxy (see the module docstring)."""
    T = N
    _, dyn, _ = real_dyn.load(t2.real, idx, T, EVERY)
    frames, depths, dests = pw_render(t2, idx, N)
    # sanity: the re-render against the published PointWorld proxy video (codec error only)
    pub = read_frames(t2.pw_videos / f"{idx}.mp4", 3)
    rerender_mae = float(np.mean([np.abs(a.astype(np.float32) - b).mean() for a, b in zip(frames, pub)]))
    cat = np.load(episode_dir(t2.rows["full"], idx) / "proxy_cat.npz")["cat"]

    def proxy_mask(t):
        pm = cv2.resize(is_robot(cat[min(t, len(cat) - 1)]).astype(np.uint8), WH, interpolation=cv2.INTER_NEAREST) > 0
        return pm | (cv2.resize(dests[t].astype(np.uint8), WH, interpolation=cv2.INTER_NEAREST) > 0)

    px = [cv2.resize(f, WH, interpolation=cv2.INTER_AREA) for f in frames]
    l1, l2 = rgb_errors(px, real_frames(idx, T, WH), dyn, proxy_mask, T)
    res = dict(idx=idx, rgb_l1=float(np.mean(l1)) if l1 else None, rgb_l2=float(np.mean(l2)) if l2 else None,
               rgb_n=len(l1), rerender_mae_first3=rerender_mae)
    f0 = t2.stereo_dir / f"{idx}_ext1.npy"
    if f0.exists():
        g0 = stereo_depth(f0)
        dl2, dl1, ql2, ql1 = [], [], [], []
        for t in DEPTH_STEPS:
            ft = t2.stereo_seq_dir / f"{idx}_ext1_t{t}.npy"
            if t >= T or t >= len(depths) or not ft.exists():
                continue
            gt = stereo_depth(ft)
            p = depths[t]
            m = change_mask(p, depths[0], gt, g0)
            if m.sum() < 100:
                continue
            e = p[m] - gt[m]
            dl2.append(float((e ** 2).mean()))
            dl1.append(float(np.abs(e).mean()))
            # the same with the proxy depth quantized as the rule-based rows' depth videos
            p = _q8(depths[t])
            m = change_mask(p, _q8(depths[0]), gt, g0)
            if m.sum() >= 100:
                e = p[m] - gt[m]
                ql2.append(float((e ** 2).mean()))
                ql1.append(float(np.abs(e).mean()))
        res.update(depth_l2=float(np.mean(dl2)) if dl2 else None, depth_l1=float(np.mean(dl1)) if dl1 else None,
                   depth_n=len(dl2), depth_l2_q8=float(np.mean(ql2)) if ql2 else None,
                   depth_l1_q8=float(np.mean(ql1)) if ql1 else None)
    return res


# ----------------------------------------------------------------------------------------------- aligned metrics
def aligned(t2, idx):
    T = N
    _, dyn, _ = real_dyn.load(t2.real, idx, T, EVERY)
    real = real_frames(idx, T, WH)
    rgb, cats, helds, dep, dests = {}, {}, {}, {}, None
    for r in t2.names:
        if r == "pw":
            _, pwdep, dests = pw_render(t2, idx, N)
            rgb[r] = read_frames(t2.pw_videos / f"{idx}.mp4", T, WH)
            dep[r] = _video_depth(pwdep)
            continue
        d = episode_dir(t2.rows[r], idx)
        rgb[r] = read_frames(d / RGB, T, WH)
        cats[r] = np.load(d / "proxy_cat.npz")["cat"]
        helds[r] = sorted(held_within(json.loads((d / "info.json").read_text()), T))
        dep[r] = decoded_depth(d / DEPTH, DEPTH_STEPS)
    acc = {f"{m}_{r}": [] for m in ("U", "R") for r in t2.names}
    npx = {"U": [], "R": []}
    for t in range(1, min([T, len(real)] + [len(v) for v in rgb.values()])):
        mag = np.zeros(WH[::-1], bool)
        prox = dyn[t - 1].copy()
        for r in t2.names:
            mag |= magenta(rgb[r][t])
            if r == "pw":
                prox |= cv2.resize(dests[t].astype(np.uint8), WH, interpolation=cv2.INTER_NEAREST) > 0
            else:
                c = cats[r][min(t, len(cats[r]) - 1)]
                m = is_robot(c) | np.isin(c, [C_OBJ0 + k for k in helds[r]])
                prox |= cv2.resize(m.astype(np.uint8), WH, interpolation=cv2.INTER_NEAREST) > 0
        for key, m in (("U", prox & ~mag), ("R", dyn[t - 1] & ~mag)):
            npx[key].append(int(m.sum()))
            if m.sum() < 50:
                continue
            g = real[t][m].astype(np.float32) / 255
            for r in t2.names:
                acc[f"{key}_{r}"].append(float(np.abs(rgb[r][t][m].astype(np.float32) / 255 - g).mean()))
    res = dict(idx=idx, **{f"rgb_l1_{k}": (float(np.mean(v)) if v else None) for k, v in acc.items()},
               rgb_px_U=float(np.mean(npx["U"])), rgb_px_R=float(np.mean(npx["R"])))
    f0 = t2.stereo_dir / f"{idx}_ext1.npy"
    if f0.exists():
        g0 = stereo_depth(f0)
        dacc = {f"{m}_{r}": [] for m in ("U", "R") for r in t2.names}
        dpx = {"U": [], "R": []}
        for t in DEPTH_STEPS:
            ft = t2.stereo_seq_dir / f"{idx}_ext1_t{t}.npy"
            if t >= T or not ft.exists() or any(t not in dep[r] for r in t2.names):
                continue
            gt = stereo_depth(ft)
            valid = valid_depth(gt)
            mv_real = np.isfinite(g0) & (np.abs(gt - g0) > CHANGE)
            mv_any = mv_real.copy()
            for r in t2.names:
                p, p0 = dep[r][t], dep[r][0]
                valid &= valid_depth(p)
                mv_any |= np.isfinite(p0) & (np.abs(p - p0) > CHANGE)
            for key, m in (("U", valid & mv_any), ("R", valid & mv_real)):
                dpx[key].append(int(m.sum()))
                if m.sum() < 100:
                    continue
                for r in t2.names:
                    dacc[f"{key}_{r}"].append(float(((dep[r][t][m] - gt[m]) ** 2).mean()))
        res.update({f"depth_l2_{k}": (float(np.mean(v)) if v else None) for k, v in dacc.items()},
                   depth_px_U=float(np.mean(dpx["U"])) if dpx["U"] else None,
                   depth_px_R=float(np.mean(dpx["R"])) if dpx["R"] else None)
    return res


def aligned_summary(t2, rows):
    """Paired statistics of the aligned metrics over the episodes where every row has a value (bootstrap 95% interval,
    2000 resamples, one numpy default_rng(0) stream); `rows`: the summary's per-row VLM grasp and automatic metrics."""
    rs = list(load_dir(t2.out / "aligned").values())
    rng = np.random.default_rng(0)

    def stat(v):
        v = np.asarray(v, float)
        b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
        return dict(mean=float(v.mean()), median=float(np.median(v)), n=len(v),
                    ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])

    out = dict(doc=ALIGNED_DOC, main_mask="U", rows={}, n_episodes=len(rs))
    for metric in ("rgb_l1", "depth_l2"):
        for key in ("U", "R"):
            ep = [r for r in rs if all(r.get(f"{metric}_{key}_{x}") is not None for x in t2.names)]
            for x in t2.names:
                e = out["rows"].setdefault(x, {})
                e[f"{metric}_{key}"] = stat([r[f"{metric}_{key}_{x}"] for r in ep])
                if x != "full":
                    dlt = [r[f"{metric}_{key}_{x}"] - r[f"{metric}_{key}_full"] for r in ep]
                    e[f"{metric}_{key}_minus_full"] = dict(stat(dlt),
                                                           row_better_frac=float(np.mean(np.asarray(dlt) < 0)))
    out["mean_pixels"] = {k: float(np.mean([r[k] for r in rs if r.get(k) is not None]))
                          for k in ("rgb_px_U", "rgb_px_R", "depth_px_U", "depth_px_R")}
    for x in t2.names:
        row, g = rows[x], out["rows"][x]
        g["grasp_vlm"] = {k: row[k] for k in ("grasp_success_vlm", "halluc_vlm", "miss_vlm", "grasp_binary_vlm")}
        g["grasp_auto"] = row["auto"]["grasp_success"] if x != "pw" else "n/a (no simulator state)"
        g["arm_iou"] = row["auto"]["arm_iou"] if x != "pw" else "identical to full (the full system's robot render)"
    for x, g in out["rows"].items():
        f = lambda d: f"{d['mean']:.4f} (med {d['median']:.4f}, n={d['n']})"  # noqa: E731
        print(x, *[f"{k} {f(g[k])}" for k in ("rgb_l1_U", "rgb_l1_R", "depth_l2_U", "depth_l2_R")])
    print(out["mean_pixels"])
    return out


# ----------------------------------------------------------------------------------------------- image object IoU
def object_iou(t2, row, idx):
    inst, dyn, n_targets = real_dyn.load(t2.real, idx, N, EVERY)
    if n_targets == 0:
        return dict(idx=idx, row=row, n_targets=0, obj_iou=None, steps={})
    ker5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    rob = {t: np.any(ms, 0) for t, ms in inst.items() if ms}
    want = set([0] + STEPS_OBJ)
    px = {t: cv2.resize(f, WH, interpolation=cv2.INTER_AREA)
          for t, f in read_indices(t2.rgb_video(row, idx), want).items()}
    ctl = {t: cv2.resize((f[..., 1] >= 40).astype(np.uint8), WH, interpolation=cv2.INTER_NEAREST)
           for t, f in read_indices(episode_dir(t2.run_of(row), idx) / CONTROL, want).items()}
    p0 = px[0].astype(np.int16)
    k7, k5 = np.ones((7, 7), np.uint8), np.ones((5, 5), np.uint8)
    steps = {}
    for t in STEPS_OBJ:
        if t not in px or t - 1 >= len(dyn):
            continue
        tgt = dyn[t - 1].copy()
        if t in rob:
            tgt &= ~(cv2.dilate(rob[t].astype(np.uint8), ker5) > 0)
        if tgt.sum() < 50:
            continue
        p = px[t]
        ch = (np.abs(p.astype(np.int16) - p0).max(-1) > DIFF) & ~(cv2.dilate(magenta(p).astype(np.uint8), k5) > 0) \
            & ~(cv2.dilate(ctl[t], k7) > 0)
        ch = cv2.morphologyEx(ch.astype(np.uint8), cv2.MORPH_OPEN, k5) > 0
        steps[t] = float((tgt & ch).sum() / max((tgt | ch).sum(), 1))
    return dict(idx=idx, row=row, n_targets=n_targets, obj_iou=float(np.mean(list(steps.values()))) if steps else None,
                steps=steps)


# ----------------------------------------------------------------------------------------------- summary
def summary(t2, auto):
    """OUT/summary.json: per row the VLM grasp metrics, the image object IoU, the automatic metrics of
    proxy_metrics.py's summary (`auto` file), PointWorld's metrics and the aligned metrics."""
    auto = json.loads(Path(auto).read_text())["variants"]
    res = dict(prompts=dict(grasp=gv.Q_GRASP, same_object=gv.Q_SAME), parse_rule=gv.PARSE_RULE,
               same_object_rule=gv.SAME_OBJECT_RULE, method_doc=__doc__, rows={})
    for row in t2.names:
        rs = gv.matched(t2.out / "vlm" / row)
        oi = load_dir(t2.out / "object_iou" / row).values()
        x = dict(grasp_success_vlm=ci([r["agree"] for r in rs]),
                 grasp_binary_vlm=ci([(r["gt_parsed"] == "none") == (r["parsed"] == "none") for r in rs]),
                 halluc_vlm=ci([r["halluc"] for r in rs if r["gt_none"]]),
                 miss_vlm=ci([r["miss"] for r in rs if not r["gt_none"]]),
                 object_iou_image=ci([o["obj_iou"] for o in oi if o["obj_iou"] is not None]))
        if row == "pw":
            pm = load_dir(t2.out / "pw_metrics").values()
            x.update({k: ci([m[k] for m in pm if m.get(k) is not None])
                      for k in ("rgb_l1", "rgb_l2", "depth_l2", "depth_l1", "depth_l2_q8", "depth_l1_q8")})
            x["rerender_mae_first3_mean"] = float(np.mean([m["rerender_mae_first3"] for m in pm])) if pm else None
            x["arm_iou"] = "identical to full (the full system's robot render)"
            x["arm_iou_value"] = auto["full"]["arm_iou"]
            x["auto"] = None
        else:
            x["auto"] = auto[row]
        res["rows"][row] = x
    if (t2.out / "aligned").exists():
        res["aligned"] = aligned_summary(t2, res["rows"])
    (t2.out / "summary.json").write_text(json.dumps(res, indent=1))
    for r, x in res["rows"].items():
        print(r, {k: (v["mean"] if isinstance(v, dict) and "mean" in v else None) for k, v in x.items()})


def main():
    p = parser(__doc__)
    p.add_argument("stage", choices=["pw-metrics", "object-iou", "aligned", "summary"])
    p.add_argument("row", nargs="?", help="object-iou: the row")
    p.add_argument("out", type=Path, help="Table 2 output directory")
    p.add_argument("--ids", help="test episode id list (per-episode stages)")
    p.add_argument("--real-dyn", type=Path, required=True, help="real_dyn.py output directory")
    p.add_argument("--row", dest="rows", action="append", required=True, metavar="NAME=RUN",
                   help="rule-based row and its proxy run (full=RUN required)")
    p.add_argument("--pw-rollouts", type=Path, help="PointWorld rollouts <idx>.npz (adds the pw row)")
    p.add_argument("--pw-videos", type=Path, help="PointWorld RGB proxy videos <idx>.mp4")
    p.add_argument("--stereo-dir", type=Path, default=STEREO_DIR, help="first-frame stereo depth (default %(default)s)")
    p.add_argument("--stereo-seq-dir", type=Path, default=STEREO_SEQ_DIR,
                   help="later-step stereo depth (default %(default)s)")
    p.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("T", "N"), help="T-th of N shards")
    p.add_argument("--auto", type=Path, help="summary: proxy_metrics.py summary JSON of the same rows")
    a = p.parse_args()
    t2 = Table2(a.out, a.real_dyn, dict(x.split("=", 1) for x in a.rows), a.pw_rollouts, a.pw_videos, a.stereo_dir,
                a.stereo_seq_dir)
    if a.stage == "summary":
        summary(t2, a.auto)
        return
    od = a.out / {"pw-metrics": "pw_metrics", "object-iou": f"object_iou/{a.row}", "aligned": "aligned"}[a.stage]
    od.mkdir(parents=True, exist_ok=True)
    fn = {"pw-metrics": lambda i: pw_metrics(t2, i), "object-iou": lambda i: object_iou(t2, a.row, i),
          "aligned": lambda i: aligned(t2, i)}[a.stage]

    def run(idx):
        (od / f"{idx}.json").write_text(json.dumps(fn(idx)))
        print("done", idx, flush=True)

    each(read_ids(a.ids, *a.shard), run,
         done=lambda idx: (od / f"{idx}.json").exists() or not (a.real_dyn / f"{idx}.npz").exists())


if __name__ == "__main__":
    main()
