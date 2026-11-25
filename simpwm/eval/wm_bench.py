"""World-model video benchmark on the DROID test set (Table 1, and the video metrics of Table 3's sim-to-real
ablation; evaluation only), for any directory of generated exterior-1 videos GEN_DIR/<idx>.mp4 whose frame k shows
recorded step k * STRIDE (1: 15 fps generations; 3: Ctrl-World's 5 Hz).

One protocol for every row of a table (--protocol): the generation and the real recording are both reduced to its size
and steps before any metric; models that need another input size (SAM 3, I3D, Depth Anything, the VLM) get these
frames resampled. table1 (baselines): 320x192 at recorded steps 0, 3, ..., 96 (5 Hz, 33 frames; Ctrl-World's output, the
coarsest compared model), robot / target masks at 640x352; table3 (our ablation, at its native output): 1280x704,
steps 0..96 (97 frames, 15 fps). Step 0 is the given first frame.

  video quality  PSNR, LPIPS (AlexNet) at the protocol size, steps > 0; FVD from I3D features (StyleGAN-V torchscript
                 I3D, 16 frames: steps 6, 12, ..., 96, 224x224), Frechet distance between the generated and real sets
                 with a delete-a-group jackknife interval
  robot IoU      SAM 3 robot instances ("robot arm", "robot", score >= 0.5) of the generated and the real frame at every
                 protocol step, each side the instances touching the reference proxy run's robot render
                 (arm_iou.pick); pixel-weighted IoU over the steps
  traj.          WorldArena trajectory accuracy (video_quality/processing/detection_tracking.py + WorldArena/
                 trajectory_accuracy.py, imported): SAM 3 boxes for "robot arm" / "end effector" / "gripper" (score
                 >= 0.3, NMS 0.3) on frames resized to 640x480, best box per image half, last valid point kept up to
                 10 frames, gaps <= 5 interpolated, window-3 smoothing; eval_traj's 1 / (DTW / path length) on the GT's
                 farthest-reaching track, normalized with WorldArena's empirical bounds (0, 40.854) to [0, 1]
  depth          Depth Anything V2 (metric indoor) on the generated frame at steps 0, 24, 48, 72, 96, scaled per step
                 by the median ratio to the real stereo depth over static pixels (real depth moved < 2 cm, no robot
                 on either side); then the proxy_metrics.py depth error: both in 0.2-3 m and the real or generated depth
                 moved > 2 cm from step 0, mean squared error in m^2
  Phys-IQ        Physics-IQ's metrics (physiq/calculate_and_write_metrics_to_csv.py, imported) between the real and the
                 generated video at the protocol's steps and size, with Physics-IQ's motion masks (running-average
                 frame difference, threshold 10, open / close 5x5), the robot (SAM 3, dilated) removed from both videos
                 and masks: spatial IoU, spatiotemporal IoU, weighted spatial IoU, MSE; the score combines them as
                 calculate_iq_score does with the physical-variance terms set to 1 (IoUs) and 0 (MSE), there being no
                 second real take: 100 * (mean of the three IoUs - MSE), clipped
  interact.      WorldArena's VLM judge (video_quality/VLM_judge.py: sample_frames' 16-frame choice applied to the
                 protocol frames, build_multi_metric_prompt with the task instruction, run_qwen_vl, vlm.py's model),
                 Interaction_Quality score / 5
  deform. IoU    episodes with a deformable object (keyword rule on the task instruction, `deformable`) in which the
                 real robot grasps a target: during each real grasp (stall step to release, grasp_gt.real_grasps), the
                 target tracked with the SAM 3 tracker from its step-0 mask in the generated and in the real video, the
                 robot subtracted from both, the generated mask moved and scaled onto the real one (centroids, sqrt of
                 the area ratio), mask IoU; mean over the grasped steps

External code under --bench-root (default deps/benchmarks): WorldArena/, physics-IQ-benchmark/, fvd/i3d_torchscript.pt,
pylib/ (fastdtw, lpips).
Stages (per-episode results cached, so shards can be resubmitted):
  export  GEN_TEST_DIR OUT_DIR       generation panels (video/eval_config.py output) -> OUT_DIR/<idx>.mp4
  real    OUT                        real-side caches (robot masks, boxes, I3D features, deformation targets)
  gen     OUT NAME GEN_DIR           all metrics but interact.
  vlm     OUT NAME GEN_DIR           interact.
  summary OUT SUMMARY_JSON NAME...   means, bootstrap 95% intervals and n over the episodes every row has
usage: python -m simpwm.eval.wm_bench export GEN_TEST_DIR OUT_DIR
       python -m simpwm.eval.wm_bench real OUT --ids IDS_TXT --ref-run RUN --test-set DS [--protocol P] [--shard T N]
       python -m simpwm.eval.wm_bench gen OUT NAME GEN_DIR --ids IDS_TXT --ref-run RUN --test-set DS [--stride S] [...]
       python -m simpwm.eval.wm_bench vlm OUT NAME GEN_DIR --ids IDS_TXT --test-set DS [--stride S] [--vlm MODEL] [...]
       python -m simpwm.eval.wm_bench summary OUT SUMMARY_JSON NAME [NAME ...] [--protocol P]
  RUN: the reference proxy run (its camera / robot render picks the robot instances and the grasp targets)
  DS:  the test video dataset (videos/<idx>.mp4 = the real recordings, dataset.jsonl captions)
"""

import json
import re
import sys
import types
from pathlib import Path

import cv2
import imageio.v2 as iio
import numpy as np
import torch

from simpwm import paths
from simpwm.eval import vlm
from simpwm.eval.common import (
    STEREO_DIR,
    STEREO_SEQ_DIR,
    each,
    episode_dir,
    parser,
    read_frames,
    read_ids,
    read_jsonl,
    read_steps,
    run_camera,
    states,
)

BENCH = paths.DEPS / "benchmarks"  # WorldArena, physics-IQ-benchmark, fvd/i3d_torchscript.pt, pylib
N = 97
PROTOCOLS = {  # every video is first reduced to `size` at `steps`; robot / target masks at `wh`
    "table1": dict(steps=list(range(0, N, 3)), size=(320, 192), wh=(640, 352)),
    "table3": dict(steps=list(range(N)), size=(1280, 704), wh=(1280, 704)),
}
W, H = 1280, 704  # depth comparison size
FVD_STEPS = list(range(6, N, 6))
DEPTH_STEPS = [24, 48, 72, 96]
CHANGE, D_NEAR, D_FAR = 0.02, 0.2, 3.0
DEFORMABLE = re.compile(r"towel|cloth|fold|napkin|shirt|clothes|sweater|sweatshirt|fabric|\brag\b|tissue|sock|jacket|"
                        r"scarf|blanket|curtain|cable|\bcord|\bwire|rope|string|garment|trousers|serviette|bandana|\bsheet")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_M = {}


def use_protocol(name):
    """Select a protocol: sets the module's STEPS, SIZE and WH."""
    global PROTOCOL, STEPS, SIZE, WH
    PROTOCOL = name
    STEPS, SIZE, WH = (PROTOCOLS[name][k] for k in ("steps", "size", "wh"))


use_protocol("table1")


class Bench:
    """Inputs of the per-episode stages: the reference proxy run, the test video dataset, the stereo depth."""

    def __init__(self, ref_run=None, test_set=None, stereo_dir=STEREO_DIR, stereo_seq_dir=STEREO_SEQ_DIR):
        self.ref_run, self.test_set = ref_run and Path(ref_run), test_set and Path(test_set)
        self.stereo_dir, self.stereo_seq_dir = Path(stereo_dir), Path(stereo_seq_dir)
        self.captions = {int(r["idx"]): r["caption"] for r in read_jsonl(self.test_set / "dataset.jsonl")} \
            if test_set else {}

    def real_path(self, idx):
        return self.test_set / "videos" / f"{idx}.mp4"

    def task(self, idx):
        c = self.captions.get(int(idx))
        return c.split("performs the task: ")[-1].split(". Realistic")[0] if c is not None else ""


def deformable(caption):
    """A deformable-object keyword in the task instruction of a caption (parenthesised examples ignored)."""
    m = re.search(r"performs the task: (.*?)\. Realistic", caption)
    return bool(DEFORMABLE.search(re.sub(r"\(.*?\)", "", (m.group(1) if m else "").lower())))


# ----------------------------------------------------------------------------------------------- external code
def _worldarena():
    """WorldArena's trajectory post-processing and scoring, imported as released (its SAM 3 / PyAV imports stubbed:
    detection runs through simpwm's SAM 3 below)."""
    if "wa" not in _M:
        vq = BENCH / "WorldArena/video_quality"
        sys.path.insert(0, str(vq))
        sys.path.append(str(BENCH / "pylib"))  # fastdtw (appended: the environment's numpy stays first)
        for name in ("av", "sam3", "sam3.model_builder", "sam3.model", "sam3.model.sam3_image_processor"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["sam3.model_builder"].build_sam3_image_model = None
        sys.modules["sam3.model.sam3_image_processor"].Sam3Processor = None
        import importlib.util

        import processing.detection_tracking as dt

        # the module file itself: the WorldArena package __init__ imports every metric's dependencies
        spec = importlib.util.spec_from_file_location("wa_trajectory_accuracy",
                                                      vq / "WorldArena/trajectory_accuracy.py")
        ta = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ta)
        _M["wa"] = (dt, ta)
    return _M["wa"]


def _physiq():
    if "piq" not in _M:
        sys.path.insert(0, str(BENCH / "physics-IQ-benchmark"))
        from physiq import calculate_and_write_metrics_to_csv as m

        _M["piq"] = m
    return _M["piq"]


# ----------------------------------------------------------------------------------------------- perception
def robot_instances(img):
    from simpwm.eval import arm_iou
    from simpwm.perception import sam3

    return [i["mask"] for i in sam3.segment_concepts(img, arm_iou.PHRASES, threshold=0.3)
            if i["score"] >= arm_iou.MIN_SCORE and i["mask"].sum() >= arm_iou.MIN_PX]


def renders(b, idx, steps):
    """The reference proxy run's robot render at WH for each step (picks which SAM 3 instances are the robot)."""
    from simpwm.core.embodiment import Embodiment
    from simpwm.eval import arm_iou

    if "emb" not in _M:
        _M["emb"] = Embodiment("franka_robotiq")
    cam, yaw = arm_iou.run_camera(episode_dir(b.ref_run, idx))
    q, g = states(idx, N)
    _M["emb"].set_gripper_yaw(yaw)
    rs = arm_iou.PosedRobot(_M["emb"], [(q[min(t, len(q) - 1)], g[min(t, len(g) - 1)]) for t in steps]).masks(
        cam.scaled(WH[1], WH[0]))
    return dict(zip(steps, rs))


def robot_masks(frames, rend):
    """{step: robot mask at WH} for frames {step: image} (any size)."""
    from simpwm.eval import arm_iou

    return {t: arm_iou.pick(robot_instances(cv2.resize(f, WH, interpolation=cv2.INTER_AREA)), rend[t])
            for t, f in frames.items()}


def boxes_track(frames):
    """WorldArena's (N, 2, 2) left / right gripper track of frames {step: image}, in step order."""
    from simpwm.perception import sam3

    dt, _ = _worldarena()
    last, miss, traj = [None, None], [0, 0], []
    for t in sorted(frames):
        img = cv2.resize(frames[t], (640, 480), interpolation=cv2.INTER_LINEAR)
        dets = sam3.segment_concepts(img, ["robot arm", "end effector", "gripper"], threshold=0.3)
        boxes, scores = dt.nms_boxes([d["box"] for d in dets], [d["score"] for d in dets], iou_threshold=0.3) \
            if dets else ([], [])
        sides = [[], []]
        for bx, s in zip(boxes, scores):
            if s < 0.3:
                continue
            xc, yc = (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2
            sides[0 if xc < 320 else 1].append((s, (xc / 640.0, yc / 480.0)))
        cur = []
        for k in range(2):
            if sides[k]:
                last[k], miss[k] = max(sides[k])[1], 0
                cur.append(last[k])
            elif last[k] is not None and miss[k] < 10:
                miss[k] += 1
                cur.append(last[k])
            else:
                miss[k] += 1
                cur.append((-1.0, -1.0))
        traj.append(cur)
    traj = np.array(traj, np.float32)
    for k in range(2):
        traj[:, k] = np.array(dt.smooth_trajectory(dt.interpolate_trajectory(traj[:, k].tolist(), max_gap=5),
                                                   window_size=3))
    return traj


def traj_score(pred, gt):
    """WorldArena eval_traj on arrays (it reads .npy files), normalized with its empirical bounds."""
    _, ta = _worldarena()
    p, ip = ta.traj_interpo_fill(pred.astype("float32"))
    g, ig = ta.traj_interpo_fill(gt.astype("float32"))
    k = ta.select_farthest_traj_index(g, ig)
    try:
        raw = float(ta.NDTW(p, g, ip, ig, k))
    except ZeroDivisionError:
        raw = float("inf")
    return raw, float(np.clip(raw / 40.8540, 0, 1))


def i3d_features(frames):
    """I3D (StyleGAN-V torchscript) features of a 16-frame clip {step: image} at FVD_STEPS."""
    if "i3d" not in _M:
        _M["i3d"] = torch.jit.load(str(BENCH / "fvd/i3d_torchscript.pt")).eval().to(DEV)
    x = np.stack([cv2.resize(frames[t], (224, 224), interpolation=cv2.INTER_AREA) for t in FVD_STEPS])
    x = torch.from_numpy(x).permute(3, 0, 1, 2)[None].float().to(DEV) / 127.5 - 1  # (1, 3, T, 224, 224) in [-1, 1]
    with torch.no_grad():
        return _M["i3d"](x, rescale=False, resize=False, return_features=True).cpu().numpy()[0]


def lpips_psnr(gen, real):
    """(PSNR, AlexNet LPIPS) of the generated vs the real frames {step: image}, steps > 0."""
    if str(BENCH / "pylib") not in sys.path:
        sys.path.append(str(BENCH / "pylib"))  # lpips (appended: the environment's numpy stays first)
    import lpips

    if "lpips" not in _M:
        _M["lpips"] = lpips.LPIPS(net="alex", verbose=False).to(DEV).eval()
    ts = [t for t in sorted(gen) if t > 0]
    g = np.stack([gen[t] for t in ts]).astype(np.float32) / 255
    r = np.stack([real[t] for t in ts]).astype(np.float32) / 255
    mse = ((g - r) ** 2).mean((1, 2, 3))

    def to(a):
        return torch.from_numpy(a).permute(0, 3, 1, 2).mul(2).sub(1).to(DEV)

    with torch.no_grad():
        lp = torch.cat([_M["lpips"](to(g[i:i + 8]), to(r[i:i + 8])).flatten() for i in range(0, len(g), 8)])
    return float(np.mean(10 * np.log10(1 / np.maximum(mse, 1e-10)))), float(lp.mean())


def frechet(a, b):
    from scipy import linalg

    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(s1 @ s2).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


# ----------------------------------------------------------------------------------------------- Physics-IQ
def motion_masks(frames, threshold=10):
    """Physics-IQ's binary_mask_generator.generate_mask on in-memory RGB frames (first mask empty)."""
    out, avg = [], None
    k = np.ones((5, 5), np.uint8)
    for f in frames:
        g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0)
        if avg is None:
            avg = g.astype("float")
            out.append(np.zeros_like(g, bool))
            continue
        cv2.accumulateWeighted(g, avg, 0.3)
        b = cv2.threshold(cv2.absdiff(g, cv2.convertScaleAbs(avg)), threshold, 255, cv2.THRESH_BINARY)[1]
        b = cv2.morphologyEx(cv2.morphologyEx(b, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        out.append(b > 127)
    return np.stack(out)


def physiq(gen, real, rob_gen, rob_real):
    """Physics-IQ metrics of the generated vs the real video {step: image}, robots removed."""
    m = _physiq()
    ts = sorted(gen)
    size = SIZE

    def rob(r, t):
        return cv2.dilate(cv2.resize(r[t].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST),
                          np.ones((5, 5), np.uint8)) > 0

    keep = np.stack([~(rob(rob_gen, t) | rob(rob_real, t)) for t in ts])
    g = np.stack([cv2.resize(gen[t], size, interpolation=cv2.INTER_AREA) for t in ts]) * keep[..., None]
    r = np.stack([cv2.resize(real[t], size, interpolation=cv2.INTER_AREA) for t in ts]) * keep[..., None]
    mg, mr = motion_masks(g) & keep, motion_masks(r) & keep
    st = float(np.mean(m.spatiotemporal_iou_per_frame(mr, mg)))
    sp = float(m.spatiotemporal_iou_per_frame(m.spatial_binary_masks(mr)[None], m.spatial_binary_masks(mg)[None])[0])
    ws = float(m.compute_weighted_spatial_iou(m.weighted_spatial_mask(mr), m.weighted_spatial_mask(mg)))
    mse = float(np.mean(m.mse_per_frame(r.astype(np.float32) / 255, g.astype(np.float32) / 255)))
    score = float(np.clip(100 * ((st + sp + ws) / 3 - mse), 0, 100))
    return dict(physiq=score, physiq_spatial_iou=sp, physiq_spatiotemporal_iou=st, physiq_weighted_spatial_iou=ws,
                physiq_mse=mse)


# ----------------------------------------------------------------------------------------------- deformation IoU
def deform_targets(b, idx):
    """(step-0 target masks at 320x180, grasp windows [(t_stall, t_release)]) of an episode with a deformable object in
    which the real robot grasps an extractable target within the clip (the same episodes for every model), else None."""
    if not deformable(b.captions.get(int(idx), "")):
        return None
    from simpwm.core.embodiment import Embodiment
    from simpwm.eval import grasp_iou
    from simpwm.eval.grasp_gt import real_grasps

    d = episode_dir(b.ref_run, idx)
    cat = np.load(d / "proxy_cat.npz")["cat"][:N]
    T, Hc, Wc = cat.shape
    frames = np.stack(read_frames(b.real_path(idx), T, (Wc, Hc)))
    q, gr = states(idx, T)
    cam, yaw = run_camera(json.loads((d / "info.json").read_text()), Hc, Wc)
    emb = Embodiment("franka_robotiq")
    emb.set_gripper_yaw(yaw)
    tg, _ = grasp_iou.gt_targets(frames, cat, cam, emb, q, gr)
    wins = [(ts, t1) for _, t1, ts in real_grasps(gr) if ts < T]
    return (tg, wins) if tg and wins else None


def track_targets(frames, masks0):
    """SAM 3 tracker of the step-0 masks over frames {step: image} (steps in order) -> {j: {step: mask at WH}}."""
    from simpwm.eval import tracks_sam3

    ts = sorted(frames)
    fr = np.stack([cv2.resize(frames[t], (320, 180), interpolation=cv2.INTER_AREA) for t in ts])
    tr = tracks_sam3.track(fr, {j: m for j, m in enumerate(masks0)})
    return {j: {t: cv2.resize(tr[j][k].astype(np.uint8), WH, interpolation=cv2.INTER_NEAREST) > 0
                for k, t in enumerate(ts)} for j in tr}


def aligned_iou(a, b):
    """IoU after moving and scaling mask a onto mask b (centroids, sqrt of the area ratio)."""
    if a.sum() < 10 or b.sum() < 10:
        return 0.0 if (a.sum() >= 10) != (b.sum() >= 10) else None
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    s = np.sqrt(len(yb) / len(ya))
    M = np.float32([[s, 0, xb.mean() - s * xa.mean()], [0, s, yb.mean() - s * ya.mean()]])
    a2 = cv2.warpAffine(a.astype(np.uint8), M, (a.shape[1], a.shape[0]), flags=cv2.INTER_NEAREST) > 0
    return float((a2 & b).sum() / max((a2 | b).sum(), 1))


# ----------------------------------------------------------------------------------------------- depth
def depth_error(b, idx, gen, rob_gen, rob_real):
    """The depth error of Depth Anything V2 on the generated frames, median-scaled to the real stereo depth."""
    from simpwm import perception

    f0 = b.stereo_dir / f"{idx}_ext1.npy"
    if not f0.exists() or 0 not in gen:
        return None

    def rs(a):
        return cv2.resize(a.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)

    def robot(m):
        return cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0

    g0 = rs(np.load(f0))
    p0 = None
    errs = []
    for t in [0] + DEPTH_STEPS:
        if t not in gen:
            continue
        p = b.stereo_seq_dir / f"{idx}_ext1_t{t}.npy"
        gt = g0 if t == 0 else (rs(np.load(p)) if p.exists() else None)
        if gt is None:
            continue
        pd = cv2.resize(perception.metric_depth(gen[t]), (W, H), interpolation=cv2.INTER_LINEAR)
        static = np.isfinite(gt) & np.isfinite(g0) & (np.abs(gt - g0) < CHANGE) & (gt > D_NEAR) & (gt < D_FAR) \
            & ~robot(rob_gen[t]) & ~robot(rob_real[t]) & (pd > 0)
        if static.sum() < 1000:
            continue
        pd = pd * float(np.median(gt[static] / pd[static]))
        if t == 0:
            p0 = pd
            continue
        ok = np.isfinite(gt) & (gt >= D_NEAR) & (gt <= D_FAR) & (pd >= D_NEAR) & (pd <= D_FAR)
        moved = (np.isfinite(g0) & (np.abs(gt - g0) > CHANGE)) | \
            ((p0 is not None) & (np.abs(pd - (p0 if p0 is not None else pd)) > CHANGE))
        m = ok & moved
        if m.sum() >= 100:
            errs.append(float(((pd[m] - gt[m]) ** 2).mean()))
    return float(np.mean(errs)) if errs else None


# ----------------------------------------------------------------------------------------------- stages
def real_stage(b, out, idx):
    f = Path(out) / "real" / f"{idx}.npz"
    steps = STEPS
    fr = read_steps(b.real_path(idx), steps, SIZE)
    rob = robot_masks(fr, renders(b, idx, steps))
    traj = boxes_track(fr)
    feat = i3d_features(fr)
    dt = deform_targets(b, idx)
    extra = {}
    if dt:
        tg, wins = dt
        tr = track_targets(fr, tg)
        extra = dict(tgt=np.packbits(np.stack([np.stack([tr[j][t] for t in steps]) for j in sorted(tr)]), axis=-1),
                     tgt0=np.packbits(np.stack(tg), axis=-1), tgt0_shape=np.array(np.stack(tg).shape),
                     wins=np.array(wins))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, robot=np.packbits(np.stack([rob[t] for t in steps]), axis=-1), traj=traj, i3d=feat,
                        tgt_checked=np.array(1), **extra)


def load_real(out, idx):
    z = np.load(Path(out) / "real" / f"{idx}.npz")

    def unpack(a, w):
        return np.unpackbits(a, axis=-1)[..., :w].astype(bool)

    r = dict(robot=dict(zip(STEPS, unpack(z["robot"], WH[0]))), traj=dict(zip(STEPS, z["traj"])), i3d=z["i3d"])
    if "tgt" in z:
        r["tgt"] = {j: dict(zip(STEPS, m)) for j, m in enumerate(unpack(z["tgt"], WH[0]))}
        r["tgt0"] = unpack(z["tgt0"], int(z["tgt0_shape"][-1]))
        r["wins"] = [tuple(w) for w in z["wins"]]
    return r


def gen_stage(b, out, name, gen_dir, stride, idx):
    f = Path(out) / name / f"{idx}.json"
    real = load_real(out, idx)
    assert all(t % stride == 0 for t in STEPS), "the protocol's steps must be frames of the generation"
    gen = read_steps(Path(gen_dir) / f"{idx}.mp4", STEPS, SIZE, stride)
    steps = sorted(gen)
    rf = read_steps(b.real_path(idx), steps, SIZE)
    res = dict(idx=idx, steps=len(steps))
    res["psnr"], res["lpips"] = lpips_psnr(gen, rf)
    rob = robot_masks(gen, renders(b, idx, steps))
    # robot IoU over the protocol's steps
    i = sum(int((rob[t] & real["robot"][t]).sum()) for t in steps)
    u = sum(int((rob[t] | real["robot"][t]).sum()) for t in steps)
    res["robot_iou"] = i / u if u else None
    # trajectory accuracy against the real track at the same steps
    res["traj_raw"], res["traj"] = traj_score(boxes_track(gen), np.stack([real["traj"][t] for t in steps]))
    res["depth_l2"] = depth_error(b, idx, gen, rob, real["robot"])
    res.update(physiq(gen, rf, rob, {t: real["robot"][t] for t in steps}))
    if "tgt" in real:  # deformation IoU
        tr = track_targets(gen, list(real["tgt0"]))
        ious = []
        for ts, t1 in real["wins"]:
            for t in steps:
                if ts <= t <= t1:
                    for j in tr:
                        v = aligned_iou(tr[j][t] & ~rob[t], real["tgt"][j][t] & ~real["robot"][t])
                        if v is not None:
                            ious.append(v)
        res["deform_iou"] = float(np.mean(ious)) if ious else None
    f.parent.mkdir(parents=True, exist_ok=True)
    if all(t in gen for t in FVD_STEPS):
        np.save(Path(out) / name / f"{idx}_i3d.npy", i3d_features(gen))
    f.write_text(json.dumps(res))


def vlm_stage(b, out, name, gen_dir, stride, idx):
    f = Path(out) / name / "vlm" / f"{idx}.json"
    if "vj" not in _M:
        sys.path.insert(0, str(BENCH / "WorldArena/video_quality"))
        import VLM_judge

        _M["vj"] = VLM_judge
    vj = _M["vj"]
    model, proc = vlm.load()
    from PIL import Image

    # VLM_judge.sample_frames' choice of 16 frames, applied to the protocol's frames
    pf = read_steps(Path(gen_dir) / f"{idx}.mp4", STEPS, SIZE, stride)
    ts = sorted(pf)
    frames = [Image.fromarray(pf[ts[int(i * len(ts) / 16)]]) for i in range(16)] if len(ts) > 16 else \
        [Image.fromarray(pf[t]) for t in ts]
    raw = vj.run_qwen_vl(model, proc, vj.build_multi_metric_prompt(b.task(idx)), frames)
    parsed = vj.safe_parse_scores(raw)
    norm = vj.normalize_metrics(parsed) if parsed else None
    iq = (norm or {}).get("Interaction_Quality", {}) if isinstance(norm, dict) else {}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(dict(idx=idx, raw=raw, interact=iq.get("score_normalized", 0.0) if iq else 0.0,
                                 interact_score=iq.get("score") if iq else None, parsed=parsed is not None)))


def export(gen_test_dir, out_dir, ids=None):
    """Generation panels [reference | generation] of video/eval_config.py shards (shard_*/samples/step_000000_<k>.mp4,
    the k-th episode of order.json) -> out_dir/<idx>.mp4 (the generated, rightmost 1280 columns); only `ids` if
    given."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sh in sorted(Path(gen_test_dir).glob("shard_*")):
        if not (sh / "order.json").exists():
            continue
        for k, idx in enumerate(json.loads((sh / "order.json").read_text()), 1):
            src, dst = sh / "samples" / f"step_000000_{k}.mp4", out_dir / f"{idx}.mp4"
            if (ids is None or idx in ids) and src.exists() and not dst.exists():
                r = iio.get_reader(str(src), "ffmpeg")
                fps = r.get_meta_data().get("fps", 15)
                iio.mimwrite(dst, [f[:, -1280:] for f in r], fps=fps, quality=9, macro_block_size=1)
                r.close()
    print(out_dir, len(list(out_dir.glob("*.mp4"))))


def summary(out, path, names):
    out = Path(out)
    rows = {n: {int(p.stem): json.loads(p.read_text()) for p in (out / n).glob("*.json")} for n in names}
    vlm_rows = {n: {int(p.stem): json.loads(p.read_text()) for p in (out / n / "vlm").glob("*.json")} for n in names}
    common = sorted(set.intersection(*[set(r) for r in rows.values()]))
    rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray([v for v in x if v is not None], float)
        if not len(x):
            return None
        bs = rng.choice(x, (2000, len(x))).mean(1)
        return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(len(x))]

    res = dict(n_common=len(common), rows={})
    keys = ["psnr", "lpips", "robot_iou", "traj", "traj_raw", "depth_l2", "physiq", "physiq_spatial_iou",
            "physiq_spatiotemporal_iou", "physiq_weighted_spatial_iou", "physiq_mse", "deform_iou"]
    for n in names:
        r = {k: ci([rows[n][i].get(k) for i in common]) for k in keys}
        r["interact"] = ci([vlm_rows[n][i]["interact"] for i in common if i in vlm_rows[n]])
        # FVD with a delete-a-group jackknife interval (20 groups of episodes): resampling with replacement duplicates
        # features and biases the Frechet distance upward, so a bootstrap interval would not cover the estimate
        ids = [i for i in common if (out / n / f"{i}_i3d.npy").exists()]
        if len(ids) > 20:
            g = np.stack([np.load(out / n / f"{i}_i3d.npy") for i in ids])
            rr = np.stack([load_real(out, i)["i3d"] for i in ids])
            fv = frechet(g, rr)
            grp = np.arange(len(ids)) % 20
            jk = np.array([frechet(g[grp != k], rr[grp != k]) for k in range(20)])
            se = float(np.sqrt(19 / 20 * ((jk - jk.mean()) ** 2).sum()))
            r["fvd"] = [fv, fv - 1.96 * se, fv + 1.96 * se, len(ids)]
        res["rows"][n] = r
    Path(path).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    global BENCH
    p = parser(__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("export", help="generation panels -> <idx>.mp4")
    s.add_argument("gen_test_dir", type=Path)
    s.add_argument("out_dir", type=Path)
    s.add_argument("--ids", help="export only these episodes")
    for stage in ("real", "gen", "vlm"):
        s = sub.add_parser(stage)
        s.add_argument("out", type=Path, help="benchmark output directory")
        if stage != "real":
            s.add_argument("name", help="row name")
            s.add_argument("gen_dir", type=Path, help="generated videos <idx>.mp4")
            s.add_argument("--stride", type=int, default=1, help="recorded steps per video frame (default 1)")
        s.add_argument("--ids", required=True, help="test episode id list")
        s.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("T", "N"), help="T-th of N shards")
        s.add_argument("--test-set", type=Path, required=True, help="test video dataset (real videos, captions)")
        if stage != "vlm":
            s.add_argument("--ref-run", type=Path, required=True, help="reference proxy run")
            s.add_argument("--stereo-dir", type=Path, default=STEREO_DIR, help="first-frame stereo depth")
            s.add_argument("--stereo-seq-dir", type=Path, default=STEREO_SEQ_DIR, help="later-step stereo depth")
        else:
            s.add_argument("--vlm", help=f"VLM judge model (default {vlm.MODEL})")
    s = sub.add_parser("summary", help="aggregate rows")
    s.add_argument("out", type=Path)
    s.add_argument("path", type=Path, help="summary JSON to write")
    s.add_argument("names", nargs="+", metavar="NAME")
    for stage, s in sub.choices.items():
        if stage != "export":
            s.add_argument("--protocol", choices=list(PROTOCOLS), default="table1", help="default %(default)s")
            s.add_argument("--bench-root", type=Path, default=BENCH,
                           help="external benchmark code (default %(default)s)")
    a = p.parse_args()
    if a.stage == "export":
        export(a.gen_test_dir, a.out_dir, set(read_ids(a.ids)) if a.ids else None)
        return
    use_protocol(a.protocol)
    BENCH = a.bench_root
    if a.stage == "summary":
        summary(a.out, a.path, a.names)
        return
    vlm.use(getattr(a, "vlm", None))
    b = Bench(getattr(a, "ref_run", None), a.test_set, getattr(a, "stereo_dir", STEREO_DIR),
              getattr(a, "stereo_seq_dir", STEREO_SEQ_DIR))
    if a.stage == "real":
        out, run = a.out / "real", lambda idx: real_stage(b, a.out, idx)
    elif a.stage == "gen":
        out, run = a.out / a.name, lambda idx: gen_stage(b, a.out, a.name, a.gen_dir, a.stride, idx)
    else:
        out, run = a.out / a.name / "vlm", lambda idx: vlm_stage(b, a.out, a.name, a.gen_dir, a.stride, idx)
    suffix = ".npz" if a.stage == "real" else ".json"

    def step(idx):
        run(idx)
        print("done", idx, flush=True)

    each(read_ids(a.ids, *a.shard), step, done=lambda idx: (out / f"{idx}{suffix}").exists())


if __name__ == "__main__":
    main()
