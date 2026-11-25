"""GR1 head (ego) camera model and the robot-mask score used to fit and check it.

Camera: T_wc(t) = FK(head_pitch_link; waist + neck joints at t) @ T_hc, with a fixed head->camera offset T_hc, focal
length and radial distortion (k1, k2); intrinsics are stored for the 1280x800 source frame and scaled to the render
size. The stored calibration (resources/embodiments/gr1_head_camera.json, fitted on robot silhouettes of calibration
episodes disjoint from the test windows) holds a dataset-level camera plus per-unit ("per_robot") and per-session
refinements; gr1/session_calib.py refines it per recording session (resources/gr1/session_cameras).

Robot masks: SAM 3 instances of PHRASES with score >= MIN_SCORE; the instances touching the rendered robot (>= TOUCH
of the instance's pixels on the render, eval/arm_iou.py's rule) form the real robot mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from simpwm.core.camera import Camera
from simpwm.paths import RESOURCES

CALIB_JSON = RESOURCES / "embodiments/gr1_head_camera.json"
CALIB_KEYS = ("T_hc", "fx", "fy", "cx", "cy", "dist")
SRC_W, SRC_H = 1280, 800

PHRASES = ["robot arm", "robot hand", "robot", "robotic arm", "mechanical hand", "black robot arm"]
MIN_SCORE, TOUCH = 0.4, 0.05


def load_calib(path=CALIB_JSON):
    return json.loads(Path(path).read_text())


def intrinsics(calib, H, W):
    s = W / SRC_W
    return np.array([[calib["fx"] * s, 0, calib["cx"] * s], [0, calib["fy"] * H / SRC_H, calib["cy"] * H / SRC_H],
                     [0, 0, 1]])


def head_camera(emb, q44, calib, H, W):
    """Head camera (H x W) for one robot state."""
    emb.set_state(q44)
    return Camera(intrinsics(calib, H, W), emb.head_pose() @ np.asarray(calib["T_hc"]), H, W, tuple(calib["dist"]))


def apply(calib, x):
    """Calibration perturbed by the 9 parameters x = [rotation vector (camera frame), translation (camera frame, m),
    log focal scale, k1, k2 (added)]."""
    T = np.asarray(calib["T_hc"], float).copy()
    R0 = T[:3, :3].copy()
    T[:3, :3] = R0 @ Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = T[:3, 3] + R0 @ x[3:6]
    c = dict(calib, T_hc=T.tolist(), fx=calib["fx"] * np.exp(x[6]), fy=calib["fy"] * np.exp(x[6]))
    if len(x) > 7:
        c["dist"] = [calib["dist"][0] + x[7], calib["dist"][1] + x[8]]
    return c


def robot_instances(frame):
    """SAM 3 robot instance masks (bool) of one RGB frame."""
    from simpwm.perception import sam3

    return [i["mask"] for i in sam3.segment_concepts(frame, PHRASES, threshold=0.3) if i["score"] >= MIN_SCORE]


def robot_mask(instances, render):
    """Union of the instances that touch the rendered robot mask `render`."""
    ours = [m for m in instances if (m & render).sum() >= TOUCH * max(m.sum(), 1)]
    return np.any(ours, 0) if ours else np.zeros(render.shape, bool)
