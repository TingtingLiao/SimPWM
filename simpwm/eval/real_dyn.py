"""Real-video masks of the first --clip-frames recorded steps, computed once per episode and shared by every proxy
variant of Table 2 (evaluation only).

Per episode, at 640x352 (the 1280x720 recording stretched as the 1280x704 proxy is):
  inst  SAM 3 robot instances (arm_iou.instances) at steps 0, EVERY, ..., N-1: each variant's arm IoU picks the ones
        touching its own render (arm_iou.score), so the expensive segmentation runs once
  dyn   the moving part of the real video at steps 1..N-1: the robot (instances touching the reference run's render)
        and the objects the real gripper grasps (grasp_iou.gt_targets, strict, from the reference run's camera; tracked
        from step 0 with the SAM 3 tracker)
Writes OUT/<idx>.npz {inst_t (K,), inst (K, H, W), dyn (N-1, H, W) bit-packed, dyn_shape, n_targets}.
usage: python -m simpwm.eval.real_dyn OUT REF_RUN --ids IDS_TXT [--shard T N] [--clip-frames N]
"""

import json
from pathlib import Path

import cv2
import numpy as np

from simpwm.eval import arm_iou, grasp_iou, tracks_sam3
from simpwm.eval.common import each, episode_dir, parser, read_ids, real_frames, run_camera, states

CLIP_FRAMES = 97
WH = arm_iou.WH


def episode(out, ref, idx, emb, n=CLIP_FRAMES):
    d = episode_dir(ref, idx)
    q, g = states(idx, n)
    T = min(n, len(q))
    # robot: instances at every step (the dynamic mask needs all of them), the arm-IoU steps kept separately
    inst = arm_iou.instances(idx, range(T))
    cam, yaw = arm_iou.run_camera(d)
    emb.set_gripper_yaw(yaw)
    rs = arm_iou.PosedRobot(emb, [(q[t], g[t]) for t in range(T)]).masks(cam.scaled(WH[1], WH[0]))
    robot = np.stack([arm_iou.pick(inst[t], r) for t, r in zip(range(T), rs)])
    # grasped objects: the real targets from the reference run's camera, tracked over the clip
    cat = np.load(d / "proxy_cat.npz")["cat"][:T]
    _, H, W = cat.shape
    frames = np.stack(real_frames(idx, T, (W, H)))
    gcam, gyaw = run_camera(json.loads((d / "info.json").read_text()), H, W)
    emb.set_gripper_yaw(gyaw)
    tg, _ = grasp_iou.gt_targets(frames, cat, gcam, emb, q[:T], g[:T])
    obj = np.zeros((T, WH[1], WH[0]), bool)
    if tg:
        tr = tracks_sam3.track(frames, {j: m for j, m in enumerate(tg)})
        for j in tr:
            for t in range(T):
                if tr[j][t] is not None and tr[j][t].any():
                    obj[t] |= cv2.resize(tr[j][t].astype(np.uint8), WH, interpolation=cv2.INTER_NEAREST) > 0
    dyn = (robot | obj)[1:]
    ts = list(range(0, T, max(8, n // 15)))
    np.savez_compressed(Path(out) / f"{idx}.npz", inst_t=np.array([t for t in ts for _ in inst[t]], np.int16),
                        inst=np.array([m for t in ts for m in inst[t]], bool).reshape(-1, WH[1], WH[0]),
                        dyn=np.packbits(dyn, axis=-1), dyn_shape=np.array(dyn.shape), n_targets=len(tg))


def load(real_dir, idx, T, every):
    """(robot instances {t: [masks]} at steps 0, every, ... < T, dyn (T-1, H, W) bool for steps 1..T-1, n_targets)."""
    z = np.load(Path(real_dir) / f"{idx}.npz")
    inst = {int(t): [] for t in range(0, T, every)}
    for t, m in zip(z["inst_t"], z["inst"]):
        inst[int(t)].append(m)
    dyn = np.unpackbits(z["dyn"], axis=-1)[..., :WH[0]].astype(bool)[:T - 1]
    return inst, dyn, int(z["n_targets"])


def main():
    p = parser(__doc__)
    p.add_argument("out", type=Path, help="output directory")
    p.add_argument("ref", type=Path, help="reference proxy run (its camera and robot render pick the real masks)")
    p.add_argument("--ids", required=True, help="episode id list")
    p.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("T", "N"), help="T-th of N shards")
    p.add_argument("--clip-frames", type=int, default=CLIP_FRAMES, help="clip length in steps (default %(default)s)")
    a = p.parse_args()
    from simpwm.core.embodiment import Embodiment

    a.out.mkdir(parents=True, exist_ok=True)
    emb = Embodiment("franka_robotiq")

    def run(idx):
        episode(a.out, a.ref, idx, emb, a.clip_frames)
        print("done", idx, flush=True)

    each(read_ids(a.ids, *a.shard), run, done=lambda idx: (a.out / f"{idx}.npz").exists())


if __name__ == "__main__":
    main()
