"""Scene state of the simulator. Every object is a point set, so rigid, articulated, deformable and granular things
share one representation; only the engine decides how points move.

World frame = robot base frame of the embodiment, gravity along -z.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# category map labels of the proxy (sim/render.py writes them; evaluation reads them without importing the renderer)
C_BG, C_SUPPORT, C_ARM, C_EFF, C_PAD, C_OBJ0 = 0, 1, 2, 3, 4, 5


def is_robot(cat):
    """Robot pixels (arm, gripper, pads). Object labels are >= C_OBJ0, so `cat >= C_ARM` would include them."""
    return (cat >= C_ARM) & (cat < C_OBJ0)


@dataclass
class Obj:
    id: int
    name: str
    pts: np.ndarray                 # (N, 3) surface samples in the world frame (visible + completed back side)
    radius: float                   # splat radius used when rasterizing (m)
    pix: int = 0                    # visible pixel count in the first frame
    attached: bool = False          # currently held by the effector
    T_rel: np.ndarray | None = None # (N, 3) object points in the effector frame while attached
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # linear velocity (free fall after release)
    pinned: bool = False            # placed by the action snap: held in place (no gravity) until first grasped
    soft: bool = False              # deformable (cloth, cable): drapes from the jaws instead of following them rigidly
    # while a soft object is held: (N, 2) per point (offset along the pad axis, distance from the jaws), and the
    # effector origin at the previous step
    drape: np.ndarray | None = None
    drape_o: np.ndarray | None = None
    rgb: np.ndarray | None = None   # (N, 3) per-point first-frame colour in [0, 1] (render.color_objects); None = grey
    carried: bool = False           # attached because it rests on the held object (config.cfg.carry_children)


@dataclass
class Background:
    depth: np.ndarray               # (H, W) z-depth of the static scene in the first camera, holes filled
    pts: np.ndarray                 # (H*W, 3) its world points
    sem: np.ndarray                 # (H, W) background category: 0 = generic, 1 = support plane
    plane: np.ndarray               # (4,) support plane n.x + d = 0 (n up)
    support_pts: np.ndarray         # (M, 3) static points objects can rest on (plane + everything above it)
    depth_hi: np.ndarray | None = None  # depth at the full recording resolution (config.cfg.hires), else None


@dataclass
class State:
    t: int
    arm_q: np.ndarray
    grip: float
    objects: list
    contacts: dict = field(default_factory=dict)  # obj id -> contact strength in [0, 1] this frame
