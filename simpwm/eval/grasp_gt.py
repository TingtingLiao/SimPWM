"""Grasp identity from the real video (evaluation only): WHEN the real gripper grasps, WHAT lies between its fingers,
and whether two first-frame masks are the same object.

  real_grasps   closure segments of the observed gripper signal in which the jaws stop short of fully closed
  object_masks  SAM 2 candidates of the thing between the fingers at the stall frame (point prompts at the FK-projected
                TCP and on rings around it, off the rendered robot)
  same_object / nested   the step-0 masks overlap (the same object) / one lies inside the other's hole-filled outline
                (container vs content: ambiguous, reported apart)
The target extraction built on these (tracking a candidate back to step 0) is grasp_iou.gt_targets.
"""

import cv2
import numpy as np

CLOSE, EMPTY, TOL = 0.12, 0.975, 0.02  # closing threshold; median closure above EMPTY = closed on nothing; stall tol
SAME_OVERLAP = 0.5  # overlap / smaller area of the two step-0 masks for "the same object"
ROBOT_MAX = 0.3  # a step-0 target overlapping the step-0 robot by more than this is a robot part (e.g. a cable the
#                  robot model lacks), not the grasped object


def real_grasps(grip):
    """[(t0, t1, t_stall)] closure segments in which the jaws stop on something."""
    segs, t = [], 0
    g = np.asarray(grip)
    while t < len(g):
        if g[t] > CLOSE:
            s = t
            while t < len(g) and g[t] > CLOSE:
                t += 1
            q = g[s:t]
            if len(q) >= 3 and np.median(q[1:]) < EMPTY:
                segs.append((s, t - 1, s + int(np.argmax(q >= q.max() - TOL))))
        t += 1
    return segs


def object_masks(frame, rob, u, v, H, W, n=8, max_frac=0.08, radii=(3, 6, 9, 12)):
    """Candidate SAM 2 masks of the thing between the fingers: prompts at the TCP (u, v), then at off-robot ring points.
    max_frac: largest object as a fraction of the image (DROID's distant cameras; a close view needs more).
    radii: prompt rings around the TCP (pixels), nearest first."""
    from simpwm import perception

    pts = [(u, v)] if not rob[v, u] else []
    for r in radii:
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            x, y = int(round(u + r * np.cos(a))), int(round(v + r * np.sin(a)))
            if 0 <= x < W and 0 <= y < H and not rob[y, x]:
                pts.append((x, y))
    seen = []
    for x, y in pts[:n]:
        m = perception.segment(frame, points=[[x, y]], labels=[1])[0] & ~rob
        if 25 <= m.sum() <= max_frac * H * W and not any((m & s).sum() > 0.8 * m.sum() for s in seen):
            seen.append(m)
            yield m, (x, y)


def same_object(a, b):
    return (a & b).sum() / max(1, min(a.sum(), b.sum())) >= SAME_OVERLAP


def _filled(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(m.shape, np.uint8)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out.astype(bool)


def nested(a, b):
    """One object lies inside the other's outline (content in a bowl, item on a plate): ambiguous, reported apart."""
    return same_object(_filled(a), _filled(b))
