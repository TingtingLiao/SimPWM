"""Action following (Table 4): counterfactual actions on 100 DROID test episodes in which the real robot picks up an
object, for Ctrl-World, DreamDojo, SimPWM (robot-only / depth-proxy / RGB-proxy reference) and the SimPWM proxy readout.

Protocol: the first 97 recorded steps (0..96), ext1, 15 fps. Every video is reduced to the recorded steps 0, 3, ..., 96
(33 frames; Ctrl-World's 5 Hz output is exactly these) at 640x352 (the full field of view squashed, as every model's
frame is); distances are reported in pixels at 1280 wide (x2).

Samples (`stalled`, `targets`, `select`): the test episodes whose observed gripper stalls on something within the first
97 steps (grasp_gt.real_grasps) and for which grasp_iou.find_targets in strict mode (target at most 4% of the image,
moves with the gripper, consistent back-track) extracts the grasped object, with the camera and robot render of the
reference proxy run. The close time t_c is the stall step of the first such grasp; episodes with t_c > TC_MAX are
dropped (no time left to see the object move), and so are those whose real recording fails the "moved" rule below (the
object not visibly lifted / carried within the 97 steps). 100 of the rest drawn with numpy default_rng(0) ->
ROOT/ids100.txt (+ selection.json; ids_pool.txt: the candidates before the real-video check).

Conditions (actions computed by droid/run_counterfactual.py prep -> ROOT/actions/<idx>.npz):
  rec      the recorded actions
  noclose  gripper command / state held at 0 (open) for every step; everything else unchanged
  offset   the end effector translated by OFFSET = 5 cm along d in the robot base frame's x-y (table) plane, every step
           from 0: d = unit(f_y, -f_x, 0) with f = the unit x-y projection of the ext1 camera's viewing axis (the
           horizontal direction perpendicular to the view, pointing to the image right). Orientation and gripper are
           unchanged (it closes as recorded, beside the object). Joint-space inputs: damped least-squares IK of the FR3
           flange seeded with the recorded joints.

Primary judge (`vlm`, VLM success judging): vlm.py's model, greedy, 8 frames at the recorded steps sampled uniformly
from t_c to 96 inclusive (the recorded t_c also for noclose), 1280 wide with each model's native aspect, one multi-image
input, with the question VLM_Q about the task's object (`objects`: the same model, text only, OBJ_PROMPT ->
objects.json); yes / no from the first word. Grasp / Halluc. = fraction of "yes". Also on the real videos (rec) as a
sanity check.
Secondary judge, the "moved" rule (`moved`, supplementary), the same for every row and the real video: the target's
step-0 mask (the strict target above) tracked with SAM 2 (tracks.py) through the 33 frames; at a step t >= t_c the
target counts as moved if its mask centroid lies more than MOVE_PX (30 px at 1280 wide) from its step-0 centroid AND
the mask is within NEAR_PX (40 px at 1280 wide) of the gripper mask. Gripper mask: the SAM 3 robot instances of that
frame (arm_iou.PHRASES, score >= 0.5) that touch the FK robot render (arm_iou.pick), cut to the render's gripper
(gripper + pads, dilated by GRIP_DIL px) - the render of the recorded AND the condition's commanded pose (union), so a
model that ignores the perturbation is judged by the same rule as one that follows it. Masks under MIN_PX count as
absent.
Readout (`readout`): the proxy run's own held state: the target is moved if a proxy object that is the same object
(grasp_gt.same_object on the step-0 masks, phantoms by their footprint) is attached in a logged state (every 5th
step) with t_c <= t < 97. `follow`: agreement of the readout with the generated videos' VLM verdict.

ROOT layout: targets/, actions/, px_<cond>/ (counterfactual proxy runs), ds_<cond>/ (build-ds), gen/,
videos/<row>_<cond>/ (generated videos), eval/<row>_<cond>/ (moved rule), eval_vlm/<row>_<cond>/ (VLM judge),
objects.json, summary.json.
usage: python -m simpwm.eval.action_follow STAGE ... --root ROOT [--ids IDS_TXT] [--ref-run RUN] [--test-set DS]
  stalled --test-ids IDS               episodes with a stalled closure -> ROOT/ids_stalled.txt
  targets [--shard T N]                strict targets of --ids (e.g. ROOT/ids_stalled.txt) -> ROOT/targets/
  select --test-ids IDS                -> ROOT/ids100.txt, selection.json (after `moved real rec real` on ids_pool.txt)
  build-ds COND RUN [--name N] [--no-robot-refs]   video dataset of COND's proxies for video/eval_config.py
  export GEN_TEST_DIR OUT_DIR          generation panels -> OUT_DIR/<idx>.mp4 (--ids only)
  moved NAME COND GEN_DIR [--stride S] [--shard T N]   "moved" rule (GEN_DIR "real": the recording)
  readout COND RUN | follow | objects | vlm NAME COND GEN_DIR [--stride S] [--shard T N] | review | summary
  figures [IDX ...]                    paper-figure grids -> ROOT/figs/
"""

import csv
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from simpwm.eval import vlm
from simpwm.eval.common import (
    centroid,
    episode_dir,
    parser,
    raw_dir,
    read_ids,
    read_indices,
    read_steps,
    real_frames,
    run_camera,
    states,
    task_texts,
    write_jpg,
)

N = 97
TC_MAX = 80
STEPS = list(range(0, N, 3))
WH = (640, 352)
SC = 1280 / WH[0]  # pixels at 1280 wide per pixel here
MOVE_PX, NEAR_PX = 30, 40
GRIP_DIL = 6
MIN_PX = 20
CONDS = ("rec", "noclose", "offset")
ROWS = [("ctrl_world", "Ctrl-World"), ("dreamdojo", "DreamDojo"), ("simpwm_robot", "SimPWM Robot only"),
        ("simpwm_depth", "SimPWM Depth proxy"), ("simpwm_rgb", "SimPWM RGB proxy"), ("readout", "SimPWM proxy readout")]
VIDEO_ROWS = ("ctrl_world", "dreamdojo", "simpwm_robot", "simpwm_rgb", "simpwm_depth")
STRIDES = {"ctrl_world": 3}  # recorded steps per video frame (default 1)
VLM_Q = "Does the robot gripper pick up / hold and lift the {obj}? Answer yes or no."
OBJ_PROMPT = ("From this robot task instruction, output only the name of the object the robot picks up or "
              "manipulates: {task}")
N_VLM = 8
GENERIC = {"task", "tasks", "object", "objects", "item", "items", "thing", "things"}
FIG_STEPS = (0, 48, 96)
_M = {}


class Experiment:
    """Paths of the experiment: ROOT, the id list, the reference proxy run, the test video dataset."""

    def __init__(self, root, ids=None, ref_run=None, test_set=None):
        self.root, self.ref_run, self.test_set = Path(root), ref_run and Path(ref_run), test_set and Path(test_set)
        self.ids_file = Path(ids) if ids else self.root / "ids100.txt"

    def ids(self):
        return read_ids(self.ids_file)

    def target(self, idx):
        """(step-0 target masks, meta {targets: [{s0, s1, ts}], grasps}) of an episode."""
        z = np.load(self.root / "targets" / f"{idx}.npz")
        return z["masks0"], json.loads(str(z["meta"]))

    def task(self, idx):
        if "tasks" not in _M:
            _M["tasks"] = task_texts(self.test_set / "dataset.jsonl")
        return _M["tasks"].get(int(idx), "")


# ----------------------------------------------------------------------------------------------- samples
def targets(ex, idx, emb):
    """Strict targets of the real video over the first N steps, with the grasp each came from, and the real grasps."""
    from simpwm.eval import grasp_iou

    d = episode_dir(ex.ref_run, idx)
    cat = np.load(d / "proxy_cat.npz")["cat"][:N]
    T, H, W = cat.shape
    frames = np.stack(real_frames(idx, T, (W, H)))
    T = min(T, len(frames))
    q, g = states(idx, T)
    cam, yaw = run_camera(json.loads((d / "info.json").read_text()), H, W)
    emb.set_gripper_yaw(yaw)
    return grasp_iou.find_targets(frames, cat, cam, emb, q, g, strict=True)


def targets_stage(ex, ids):
    from simpwm.core.embodiment import Embodiment

    emb = Embodiment("franka_robotiq")
    (ex.root / "targets").mkdir(parents=True, exist_ok=True)
    for idx in ids:
        f = ex.root / "targets" / f"{idx}.npz"
        if f.exists():
            continue
        try:
            tg, grasps = targets(ex, idx, emb)
        except Exception as e:  # one bad episode must not stop the shard
            print("FAILED", idx, repr(e), flush=True)
            continue
        np.savez_compressed(f, masks0=np.stack([t["m0"] for t in tg]) if tg else np.zeros((0, 180, 320), bool),
                            meta=json.dumps(dict(targets=[{k: v for k, v in t.items() if k != "m0"} for t in tg],
                                                 grasps=[list(map(int, x)) for x in grasps])))
        print(idx, len(grasps), "grasps", [(t["s0"], t["ts"]) for t in tg], flush=True)


def stalled_ids(test_ids):
    """Test episodes whose observed gripper stalls on something within the first N steps (target extraction runs on
    these only)."""
    from simpwm.eval.grasp_gt import real_grasps

    return [i for i in test_ids if real_grasps(states(i, N)[1])]


def select(ex, test_ids):
    cand, why = [], {}
    for i in test_ids:
        if not (ex.root / "targets" / f"{i}.npz").exists():
            why[i] = "no stalled closure in 97 steps (not run)"
            continue
        _, meta = ex.target(i)
        if not meta["targets"]:
            why[i] = "no strict target"
        elif meta["targets"][0]["ts"] > TC_MAX:
            why[i] = f"close after step {TC_MAX}"
        else:
            cand.append(i)
    (ex.root / "ids_pool.txt").write_text("\n".join(map(str, cand)) + "\n")
    # the real robot must visibly move the target within the clip: the "moved" rule on the real recording
    real = {i: json.loads(p.read_text())["moved"] for i in cand
            if (p := ex.root / "eval" / "real_rec" / f"{i}.json").exists()}
    if len(real) < len(cand):
        print(f"real-video check missing for {len(cand) - len(real)} of {len(cand)} candidates "
              "(run `moved real rec real --ids ROOT/ids_pool.txt` first)")
        return
    pool = [i for i in cand if real[i]]
    rng = np.random.default_rng(0)
    pick = sorted(int(x) for x in rng.choice(pool, size=min(100, len(pool)), replace=False))
    (ex.root / "ids100.txt").write_text("\n".join(map(str, pick)) + "\n")
    tc = {i: ex.target(i)[1]["targets"][0]["ts"] for i in pick}
    sel = dict(rule=(
        "test episodes -> candidates: the observed gripper stalls on something within the first 97 steps "
        "(grasp_gt.real_grasps) AND grasp_iou.find_targets in strict mode (the reference run's camera and robot "
        "render) extracts the grasped object from the real video for one of those grasps (object <= 4% of the image, "
        "moves with the gripper, consistent back-track to step 0) AND the first such grasp's stall step "
        f"t_c <= {TC_MAX} "
        "AND the real recording passes the 'moved' rule (the target visibly lifted / carried by the gripper after t_c "
        "within the 97 steps). 100 drawn with numpy.random.default_rng(0).choice(pool, 100, replace=False), sorted."),
        n_test=len(test_ids), n_stalled=sum(1 for i in test_ids if (ex.root / "targets" / f"{i}.npz").exists()),
        n_candidates=len(cand), n_real_moved=len(pool), real_moved_rate_on_candidates=len(pool) / len(cand),
        excluded={k: sum(v == k for v in why.values()) for k in set(why.values())},
        candidates=cand, pool=pool, ids=pick, t_close=tc)
    (ex.root / "selection.json").write_text(json.dumps(sel, indent=1))
    print(len(cand), "candidates,", len(pool), "real moved ->", len(pick), sel["excluded"], "t_c median",
          np.median(list(tc.values())))


# ----------------------------------------------------------------------------------------------- moved rule
def _emb():
    if "emb" not in _M:
        from simpwm.core.embodiment import Embodiment

        _M["emb"] = Embodiment("franka_robotiq")
    return _M["emb"]


def cond_states(ex, idx, cond):
    """Observed joints / gripper the condition commands, steps 0..N-1."""
    if cond == "rec":
        return states(idx, N)
    z = np.load(ex.root / "actions" / f"{idx}.npz")
    return z[f"{cond}_obs_q"][:N], z[f"{cond}_obs_g"][:N]


def gripper_regions(ex, idx, cond, steps):
    """{t: (robot render, gripper-part render dilated)} at WH: union of the recorded and the condition's pose."""
    import torch

    from simpwm.core.raster_gpu import PosedRobot
    from simpwm.eval import arm_iou

    emb = _emb()
    cam, yaw = arm_iou.run_camera(episode_dir(ex.ref_run, idx))
    c = cam.scaled(WH[1], WH[0])
    emb.set_gripper_yaw(yaw)
    rob, grip = {t: np.zeros(WH[::-1], bool) for t in steps}, {t: np.zeros(WH[::-1], bool) for t in steps}
    for cc in {"rec", cond}:
        q, g = cond_states(ex, idx, cc)
        pr = PosedRobot(emb, [(q[min(t, len(q) - 1)], g[min(t, len(g) - 1)]) for t in steps])
        parts = pr.gr.parts(pr.V, c).cpu().numpy() if pr.V is not None else None
        for k, t in enumerate(steps):
            if parts is not None:
                p = parts[k]
            else:  # CPU fallback
                emb.set_state(q[t], g[t])
                p = emb.render(c)[1]
            rob[t] |= p >= 0
            grip[t] |= p >= 1
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    k = np.ones((2 * GRIP_DIL + 1, 2 * GRIP_DIL + 1), np.uint8)
    return {t: (rob[t], cv2.dilate(grip[t].astype(np.uint8), k).astype(bool)) for t in steps}


def moved_episode(ex, idx, cond, frames):
    """frames {step: image at WH} -> per-step record and the moved flag."""
    from simpwm.eval import arm_iou, tracks
    from simpwm.eval.wm_bench import robot_instances

    masks0, meta = ex.target(idx)
    tc = meta["targets"][0]["ts"]
    m0 = cv2.resize(masks0[0].astype(np.uint8), WH, interpolation=cv2.INTER_NEAREST) > 0
    steps = sorted(frames)
    assert steps == STEPS, (idx, len(steps))
    tr = tracks.track(np.stack([frames[t] for t in steps]), {0: m0})[0]
    c0 = centroid(m0)
    late = [t for t in steps if t >= tc]
    reg = None
    per, moved, first = {}, False, None
    for k, t in enumerate(steps):
        if t < tc:
            continue
        m = tr[k]
        if m.sum() < MIN_PX:
            per[t] = dict(present=False)
            continue
        disp = float(np.linalg.norm(centroid(m) - c0) * SC)
        rec = dict(present=True, disp=round(disp, 1))
        if disp > MOVE_PX:  # the gripper mask only where the displacement test passes (SAM 3 is the slow part)
            if reg is None:
                reg = gripper_regions(ex, idx, cond, late)
            rend, gzone = reg[t]
            robot = arm_iou.pick(robot_instances(frames[t]), rend)
            gm = robot & gzone
            if gm.sum() >= MIN_PX:
                dist = cv2.distanceTransform((~gm).astype(np.uint8), cv2.DIST_L2, 5)
                near = float(dist[m].min() * SC)
            else:
                near = None
            rec.update(grip_px=int(gm.sum()), near=None if near is None else round(near, 1))
            if near is not None and near <= NEAR_PX:
                moved = True
                first = t if first is None else first
        per[t] = rec
    return dict(idx=idx, cond=cond, t_close=tc, moved=moved, first_moved=first, max_disp=max(
        [r["disp"] for r in per.values() if r.get("present")], default=None), steps=per)


def _source(gen_dir, idx):
    return raw_dir(idx) / "ext1.mp4" if str(gen_dir) == "real" else Path(gen_dir) / f"{idx}.mp4"


def moved_stage(ex, name, cond, gen_dir, stride, ids):
    out = ex.root / "eval" / f"{name}_{cond}"
    out.mkdir(parents=True, exist_ok=True)
    for idx in ids:
        f = out / f"{idx}.json"
        if f.exists():
            continue
        src = _source(gen_dir, idx)
        if not src.exists():
            print("missing", src, flush=True)
            continue
        try:
            r = moved_episode(ex, idx, cond, read_steps(src, STEPS, WH, stride))
        except Exception as e:
            import traceback

            print("FAILED", idx, repr(e), traceback.format_exc(), flush=True)
            continue
        r["video"] = str(src)
        f.write_text(json.dumps(r))
        print(name, cond, idx, "moved" if r["moved"] else "-", r["first_moved"], r["max_disp"], flush=True)


# ----------------------------------------------------------------------------------------------- proxy readout
def readout(ex, cond, run):
    from simpwm.eval.grasp_gt import same_object

    out = ex.root / "eval" / f"readout_{cond}"
    out.mkdir(parents=True, exist_ok=True)
    for idx in ex.ids():
        d = episode_dir(run, idx)
        if not (d / "info.json").exists():
            print("missing", d, flush=True)
            continue
        info = json.loads((d / "info.json").read_text())
        z = np.load(d / "proxy_cat.npz")
        masks0, meta = ex.target(idx)
        tgt, tc = masks0[0], meta["targets"][0]["ts"]
        objs = {int(k): m for k, m in zip(z["obj_ids"], z["masks0"])}
        objs.update({int(k): m for k, m in zip(z["phantom_ids"], z["phantom_masks0"])})
        same = sorted(k for k, m in objs.items() if same_object(m, tgt))
        att = sorted({int(o["id"]) for s in info.get("states", []) if tc <= s["t"] < N for o in s["objects"]
                      if o.get("attached")})
        att_any = sorted({int(o["id"]) for s in info.get("states", []) if s["t"] < N for o in s["objects"]
                          if o.get("attached")})
        r = dict(idx=idx, cond=cond, t_close=tc, target_objects=same, attached_after_close=att, attached_97=att_any,
                 moved=bool(set(same) & set(att)), held_other=bool(set(att) - set(same)), run=str(d))
        (out / f"{idx}.json").write_text(json.dumps(r))
    rs = [json.loads(p.read_text()) for p in out.glob("*.json")]
    print(cond, len(rs), "moved", sum(r["moved"] for r in rs), "target matched",
          sum(bool(r["target_objects"]) for r in rs))


def follow_readout(ex):
    """Agreement of the proxy readout (simulator: target held after t_c) with the generated video's VLM verdict, per SimPWM
    row and condition, over the episodes both exist for -> ROOT/follow_readout.json (bootstrap 95% interval, 2000
    resamples, numpy default_rng(0))."""
    rng = np.random.default_rng(0)
    out = dict(note=follow_readout.__doc__, rows={})
    for name in ("simpwm_rgb", "simpwm_depth", "simpwm_robot"):
        out["rows"][name] = {}
        for c in CONDS:
            pairs = []
            for i in ex.ids():
                a = ex.root / "eval" / f"readout_{c}" / f"{i}.json"
                b = ex.root / "eval_vlm" / f"{name}_{c}" / f"{i}.json"
                if a.exists() and b.exists():
                    pairs.append((i, bool(json.loads(a.read_text())["moved"]),
                                  bool(json.loads(b.read_text())["moved"])))
            if not pairs:
                continue
            agree = np.array([r == g for _, r, g in pairs], float)
            bs = rng.choice(agree, (2000, len(agree))).mean(1)

            def cnt(r, g):
                return sum(1 for _, x, y in pairs if x == r and y == g)

            out["rows"][name][c] = dict(
                n=len(pairs), agreement=float(agree.mean()),
                ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                counts={"readout_yes_gen_yes": cnt(True, True), "readout_yes_gen_no": cnt(True, False),
                        "readout_no_gen_yes": cnt(False, True), "readout_no_gen_no": cnt(False, False)},
                readout_no_gen_yes=cnt(False, True),
                readout_no_gen_yes_ids=[i for i, x, y in pairs if not x and y])
            r = out["rows"][name][c]
            print(name, c, r["n"], round(r["agreement"], 3), [round(x, 3) for x in r["ci95"]], r["counts"])
    (ex.root / "follow_readout.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------------------------- VLM judge (primary)
def _obj_name(raw):
    """The first line of the model's answer (it lists several objects for multi-object tasks), lower case; a generic
    answer ("task", ...) becomes "object"."""
    o = next((line for line in raw.splitlines() if line.strip()), "").strip().strip(".").strip('"').strip("*")
    o = o.strip().lower()
    return "object" if o in GENERIC or not o else o


def objects_stage(ex):
    f = ex.root / "objects.json"
    if f.exists():  # re-derive the names from the stored raw answers (post-processing changes)
        out = json.loads(f.read_text())
        for v in out["objects"].values():
            v["object"] = _obj_name(v["raw"])
        f.write_text(json.dumps(out, indent=1))
    out = json.loads(f.read_text()) if f.exists() else dict(prompt=OBJ_PROMPT, model=vlm.name(), objects={})
    for i in ex.ids():
        if str(i) in out["objects"]:
            continue
        t = ex.task(i)
        raw = vlm.ask([{"type": "text", "text": OBJ_PROMPT.format(task=t)}], 24)
        out["objects"][str(i)] = dict(task=t, object=_obj_name(raw), raw=raw)
        print(i, t, "->", out["objects"][str(i)]["object"], flush=True)
    f.write_text(json.dumps(out, indent=1))


def vlm_steps(tc):
    """The N_VLM recorded steps sampled uniformly from t_c to 96 inclusive."""
    return [int(round(x)) for x in np.linspace(tc, N - 1, N_VLM)]


def vlm_frames(src, stride, steps):
    """The frames at the recorded steps (frame round(step / stride): Ctrl-World's 5 Hz frames are steps 0, 3, ...),
    resized to 1280 wide keeping the native aspect (320x192 -> 1280x768, 640x480 -> 1280x960, 1280x704 / 720 as is)."""
    got = read_indices(src, {int(round(t / stride)) for t in steps})
    last = max(got)
    out = []
    for t in steps:
        f = got.get(int(round(t / stride)), got[last])
        h, w = f.shape[:2]
        out.append(f if w == 1280 else cv2.resize(f, (1280, int(round(h * 1280 / w))), interpolation=cv2.INTER_CUBIC))
    return out


def vlm_stage(ex, name, cond, gen_dir, stride, ids):
    from PIL import Image

    objs = json.loads((ex.root / "objects.json").read_text())["objects"]
    out = ex.root / "eval_vlm" / f"{name}_{cond}"
    out.mkdir(parents=True, exist_ok=True)
    for idx in ids:
        f = out / f"{idx}.json"
        q = VLM_Q.format(obj=objs[str(idx)]["object"])
        if f.exists() and json.loads(f.read_text())["question"] == q:
            continue
        src = _source(gen_dir, idx)
        if not src.exists():
            print("missing", src, flush=True)
            continue
        tc = ex.target(idx)[1]["targets"][0]["ts"]
        steps = vlm_steps(tc)
        frames = vlm_frames(src, stride, steps)
        raw = vlm.ask([*[{"type": "image", "image": Image.fromarray(x)} for x in frames],
                       {"type": "text", "text": q}], 16)
        w = re.findall(r"[a-z]+", raw.lower())
        ans = w[0] if w else ""
        r = dict(idx=idx, row=name, cond=cond, t_close=tc, steps=steps, frame_size=list(frames[0].shape[:2]),
                 object=objs[str(idx)]["object"], question=q, raw=raw, answer=ans, moved=ans == "yes",
                 parsed=ans in ("yes", "no"), video=str(src))
        f.write_text(json.dumps(r))
        print(name, cond, idx, repr(raw), flush=True)


def review_set(ex):
    """60 generated videos (5 video rows x 3 conditions x 4, numpy default_rng(0)) for a blind human check: the 8
    frames the VLM saw as a jpg grid (no answer on it), a json with both judges, index.csv."""
    rng = np.random.default_rng(0)
    out = ex.root / "human_review"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in VIDEO_ROWS:
        for c in CONDS:
            have = [i for i in ex.ids() if (ex.root / "eval_vlm" / f"{name}_{c}" / f"{i}.json").exists()]
            for i in sorted(int(x) for x in rng.choice(have, size=min(4, len(have)), replace=False)):
                v = json.loads((ex.root / "eval_vlm" / f"{name}_{c}" / f"{i}.json").read_text())
                sp = ex.root / "eval" / f"{name}_{c}" / f"{i}.json"
                sm = json.loads(sp.read_text())["moved"] if sp.exists() else None
                fr = vlm_frames(v["video"], STRIDES.get(name, 1), v["steps"])
                th = [cv2.putText(cv2.resize(x, (640, int(round(x.shape[0] * 640 / x.shape[1])))), f"step {t}", (8, 28),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2) for x, t in zip(fr, v["steps"])]
                stem = f"{name}_{c}_{i}"
                write_jpg(out / f"{stem}.jpg",
                          np.concatenate([np.concatenate(th[:4], 1), np.concatenate(th[4:], 1)], 0))
                rec = dict(file=f"{stem}.jpg", row=name, condition=c, idx=i, object=v["object"], question=v["question"],
                           steps=v["steps"], vlm_raw=v["raw"], vlm_answer=v["answer"], sam2_moved=sm, video=v["video"])
                (out / f"{stem}.json").write_text(json.dumps(rec, indent=1))
                rows.append(rec)
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "row", "condition", "idx", "object", "vlm_answer", "sam2_moved", "human_answer"])
        for r in rows:
            w.writerow([r[k] for k in ("file", "row", "condition", "idx", "object", "vlm_answer", "sam2_moved")] + [""])
    print(len(rows), "review items ->", out)


# ----------------------------------------------------------------------------------------------- summary
def summary(ex):
    ids = ex.ids()

    def rate(name, cond, judge="sam2"):
        d = ex.root / ("eval" if judge == "sam2" else "eval_vlm") / f"{name}_{cond}"
        rs = {i: json.loads((d / f"{i}.json").read_text()) for i in ids if (d / f"{i}.json").exists()}
        return (sum(r["moved"] for r in rs.values()) / len(rs) if rs else None), len(rs)

    out = dict(protocol=__doc__.split("ROOT layout:")[0].strip(), vlm_judge=dict(
        model=vlm.name(), decoding="greedy (do_sample=False)", question=VLM_Q, frames=(
            f"{N_VLM} recorded steps uniformly from t_c to 96 inclusive (recorded t_c also for noclose), 1280 wide, "
            "native aspect"), object_prompt=OBJ_PROMPT, objects="objects.json"), ids=ids, rows={})
    sel = json.loads((ex.root / "selection.json").read_text())
    s, n = rate("real", "rec")
    sv, nv = rate("real", "rec", "vlm")
    out["real_sanity"] = dict(vlm=dict(yes_rate=sv, n=nv), sam2=dict(
        moved_rate=s, n=n, note="1.0 by construction on the 100 (the selection requires it); before that filter:",
        candidates_moved_rate=sel["real_moved_rate_on_candidates"], n_candidates=sel["n_candidates"],
        n_candidates_moved=sel["n_real_moved"]))
    out["selection_rule"] = sel["rule"]
    for key, label in ROWS:
        js = {}
        for judge in (("sam2",) if key == "readout" else ("vlm", "sam2")):
            g, ng = rate(key, "rec", judge)
            h1, n1 = rate(key, "noclose", judge)
            h2, n2 = rate(key, "offset", judge)
            js["simulator" if key == "readout" else judge] = dict(grasp=g, n_grasp=ng, halluc_noclose=h1, n_noclose=n1,
                                                                  halluc_offset=h2, n_offset=n2)
        out["rows"][key] = dict(label=label, judges=js)
    ik = ex.root / "ik_residuals.json"
    if ik.exists():
        out["ik_residuals"] = json.loads(ik.read_text())
    (ex.root / "summary.json").write_text(json.dumps(out, indent=1))

    def f(x):
        return "  -  " if x is None else f"{100 * x:5.1f}"

    print(f"real-video sanity: VLM {f(sv)}% yes (n={nv}); SAM2 rule {f(s)}% (n={n})")
    print(f"{'row':20s} {'judge':9s} {'Grasp':>6s} {'NoClose':>8s} {'Offset':>7s}  n")
    for key, r in out["rows"].items():
        for j, x in r["judges"].items():
            print(f"{r['label']:20s} {j:9s} {f(x['grasp']):>6s} {f(x['halluc_noclose']):>8s} "
                  f"{f(x['halluc_offset']):>7s}  "
                  f"{x['n_grasp']}/{x['n_noclose']}/{x['n_offset']}")


# ----------------------------------------------------------------------------------------------- SimPWM inputs / outputs
def build_ds(ex, cond, run, name=None, robot_refs=True):
    """Video dataset ROOT/ds_COND (or ROOT/NAME) of the test set but with COND's proxies, for video/eval_config.py:
    refs/<idx>.mp4 -> RUN's ref_rgb_1280x704.mp4 (RGB proxy), refs_depth/<idx>.mp4 -> RUN's ref_depth_1280x704.mp4
    (depth proxy), refs_robot/<idx>.mp4 (video/robot_refs.py from RUN's control video), first/ and videos/ -> the test
    set's, dataset.jsonl = the test set's rows of the episodes."""
    from simpwm.video import robot_refs
    from simpwm.video.common import link, read_jsonl

    src, ds = ex.test_set, ex.root / (name or f"ds_{cond}")
    for sub in ("refs", "refs_depth", "first", "videos"):
        (ds / sub).mkdir(parents=True, exist_ok=True)
    rows = {r["idx"]: r for r in read_jsonl(src / "dataset.jsonl")}
    with open(ds / "dataset.jsonl", "w") as f:
        for i in ex.ids():
            d = episode_dir(run, i)
            if not (d / "info.json").exists():
                print("no proxy", i, flush=True)
                continue
            f.write(json.dumps(rows[i]) + "\n")
            for sub, s in (("refs", d / "conditions/ref_rgb_1280x704.mp4"),
                           ("refs_depth", d / "conditions/ref_depth_1280x704.mp4"),
                           ("first", src / "first" / f"{i}.png"), ("videos", src / "videos" / f"{i}.mp4")):
                link(s.resolve(), ds / sub / (f"{i}.png" if sub == "first" else f"{i}.mp4"))
    if robot_refs:  # robot-only references (only the robot-only model needs them)
        robot_refs.build(run, ds, 97)


def _frames_at(path, steps, wh=WH):
    got = read_indices(path, steps)
    return [cv2.resize(got[t], wh, interpolation=cv2.INTER_AREA) for t in steps]


def figures(ex, ids=None, k=3):
    """Paper-figure grids ROOT/figs/<idx>.jpg: rows real | RGB proxy, SimPWM RGB proxy, SimPWM robot-only (recorded) | the
    same (noclose), columns FIG_STEPS; the proxies (RGB proxy and robot-only reference videos, both conditions) are
    copied next to it. Default episodes: the RGB-proxy SimPWM grasps under rec and not under noclose while the robot-only
    SimPWM still moves the object under noclose (ranked by the robot-only max displacement), else any with rec grasped."""
    def ev(n, c, i):
        p = ex.root / "eval" / f"{n}_{c}" / f"{i}.json"
        return json.loads(p.read_text()) if p.exists() else None

    if not ids:
        sc = []
        for i in ex.ids():
            r = [ev(n, c, i) for n, c in (("simpwm_rgb", "rec"), ("simpwm_rgb", "noclose"), ("simpwm_robot", "rec"),
                                          ("simpwm_robot", "noclose"))]
            if None in r:
                continue
            s = 4 * r[0]["moved"] + 2 * (not r[1]["moved"]) + 2 * r[3]["moved"] + r[2]["moved"]
            sc.append((s, r[3]["max_disp"] or 0, i))
        ids = [i for _, _, i in sorted(sc, reverse=True)[:k]]
    out = ex.root / "figs"
    out.mkdir(parents=True, exist_ok=True)

    def pad(im, txt):
        return cv2.putText(np.concatenate([np.full((28, im.shape[1], 3), 255, np.uint8), im]), txt, (6, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    for i in ids:
        rows = [("real (recorded)", raw_dir(i) / "ext1.mp4")]
        for c, run in (("rec", ex.ref_run), ("noclose", ex.root / "px_noclose")):
            prox = episode_dir(run, i) / "conditions" / "ref_rgb_1280x704.mp4"
            rows += [(f"RGB proxy ({c})", prox),
                     (f"SimPWM RGB proxy ({c})", ex.root / "videos" / f"simpwm_rgb_{c}" / f"{i}.mp4"),
                     (f"SimPWM robot only ({c})", ex.root / "videos" / f"simpwm_robot_{c}" / f"{i}.mp4")]
            d = out / f"{i}_proxies"
            d.mkdir(exist_ok=True)
            shutil.copy(prox, d / f"ref_rgb_{c}.mp4")
            rr = (ex.test_set if c == "rec" else ex.root / f"ds_{c}") / "refs_robot" / f"{i}.mp4"
            if rr.exists():
                shutil.copy(rr, d / f"ref_robot_{c}.mp4")
        rows_img = []
        for label, p in rows:
            if not Path(p).exists():
                print("missing", p)
                continue
            fr = _frames_at(p, FIG_STEPS)
            rows_img.append(np.concatenate([pad(f, f"{label}  step {t}") for f, t in zip(fr, FIG_STEPS)], 1))
        write_jpg(out / f"{i}.jpg", np.concatenate(rows_img, 0), 92)
        print("figure", out / f"{i}.jpg")


def main():
    from simpwm.eval import wm_bench

    p = parser(__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    stages = {
        "stalled": [("--test-ids", dict(required=True, help="test episode id list"))],
        "targets": [], "select": [("--test-ids", dict(required=True, help="test episode id list"))],
        "build-ds": [("cond", {}), ("run", dict(type=Path, help="proxy run of the condition")),
                     ("--name", dict(help="dataset directory name under ROOT (default ds_COND)")),
                     ("--no-robot-refs", dict(action="store_true", help="skip the robot-only reference videos"))],
        "export": [("gen_test_dir", dict(type=Path)), ("out_dir", dict(type=Path))],
        "moved": [("name", {}), ("cond", {}), ("gen_dir", dict(help="generated videos <idx>.mp4, or 'real'"))],
        "vlm": [("name", {}), ("cond", {}), ("gen_dir", dict(help="generated videos <idx>.mp4, or 'real'"))],
        "readout": [("cond", {}), ("run", dict(type=Path, help="proxy run of the condition"))],
        "follow": [], "objects": [], "review": [], "summary": [],
        "figures": [("idxs", dict(nargs="*", type=int, metavar="IDX"))],
    }
    for stage, args in stages.items():
        s = sub.add_parser(stage)
        for name, kw in args:
            s.add_argument(name, **kw)
        s.add_argument("--root", type=Path, required=True, help="experiment directory")
        s.add_argument("--ids", help="episode id list (default ROOT/ids100.txt)")
        s.add_argument("--ref-run", type=Path, help="reference proxy run (recorded actions)")
        s.add_argument("--test-set", type=Path, help="test video dataset (captions, first frames, real videos)")
        if stage in ("targets", "moved", "vlm"):
            s.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("T", "N"), help="T-th of N shards")
        if stage in ("moved", "vlm"):
            s.add_argument("--stride", type=int, default=1, help="recorded steps per video frame (default 1)")
        if stage in ("objects", "vlm", "summary"):
            s.add_argument("--vlm", help=f"VLM judge model (default {vlm.MODEL})")
    a = p.parse_args()
    vlm.use(getattr(a, "vlm", None))
    ex = Experiment(a.root, a.ids, a.ref_run, a.test_set)
    shard = getattr(a, "shard", (0, 1))
    if a.stage == "stalled":
        ids = stalled_ids(read_ids(a.test_ids))
        (ex.root / "ids_stalled.txt").write_text("\n".join(map(str, ids)) + "\n")
        print(len(ids), "stalled")
    elif a.stage == "targets":
        targets_stage(ex, read_ids(ex.ids_file, *shard))
    elif a.stage == "select":
        select(ex, read_ids(a.test_ids))
    elif a.stage == "build-ds":
        build_ds(ex, a.cond, a.run, a.name, not a.no_robot_refs)
    elif a.stage == "export":
        wm_bench.export(a.gen_test_dir, a.out_dir, set(ex.ids()))
    elif a.stage == "moved":
        moved_stage(ex, a.name, a.cond, a.gen_dir, a.stride, read_ids(ex.ids_file, *shard))
    elif a.stage == "vlm":
        vlm_stage(ex, a.name, a.cond, a.gen_dir, a.stride, read_ids(ex.ids_file, *shard))
    elif a.stage == "readout":
        readout(ex, a.cond, a.run)
    elif a.stage == "follow":
        follow_readout(ex)
    elif a.stage == "objects":
        objects_stage(ex)
    elif a.stage == "review":
        review_set(ex)
    elif a.stage == "summary":
        summary(ex)
    else:
        figures(ex, a.idxs or None)


if __name__ == "__main__":
    main()
