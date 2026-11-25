"""Fourier GR1 humanoid (GR1T1 + Fourier 6-DoF hands) as a MuJoCo model for forward kinematics and robot rendering.

  emb.set_state(q44)    one 44-dim GR1 state vector (GR00T "gr1_unified" layout, radians)
  emb.head_pose()       4x4 world pose of head_pitch_link (the head camera hangs off it, see gr1/calib.py)
  emb.geom_part         per-geom part id (PART_ARM / PART_GRIPPER / PART_PAD) for core/raster_gpu.GpuRaster

The model is built directly from the URDF with MjSpec: GR1's URDF uses relative mesh paths, mimic joints and per-link
marker spheres that MuJoCo's URDF reader handles poorly. Only the upper body is built (the legs are out of the head
camera's view and fixed in the data); the root is base_link (pelvis) = the world frame.

44-dim layout: left_arm 0:7, left_hand 7:13, left_leg 13:19, neck 19:22, right_arm 22:29, right_hand 29:35,
right_leg 35:41, waist 41:44. Hand values (6 per hand, >= 0, radians of closure) are [pinky, ring, middle, index,
thumb_pitch, thumb_yaw]; the URDF finger joints close towards negative angles, so finger_proximal = -v,
thumb_yaw = -v, thumb_pitch = +v, and the URDF mimic relations give the passive joints (intermediate = 0.975 *
proximal, thumb_distal = 1.06 * thumb_pitch): 6 actuated values -> 11 joints per hand.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from simpwm.core.embodiment import PART_ARM, PART_GRIPPER, PART_PAD
from simpwm.paths import ASSETS

URDF = ASSETS / "grx_models/GRX/GR1/gr1t1/urdf/gr1t1_fourier_hand_6dof.urdf"

ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_roll",
              "wrist_pitch"]
FINGERS = ["pinky", "ring", "middle", "index"]
SL = {"left": slice(0, 7), "left_hand": slice(7, 13), "neck": slice(19, 22), "right": slice(22, 29),
      "right_hand": slice(29, 35), "waist": slice(41, 44)}
NECK = ["head_yaw_joint", "head_roll_joint", "head_pitch_joint"]
WAIST = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
LEG_PREFIX = ("left_thigh", "left_shank", "left_foot", "right_thigh", "right_shank", "right_foot")
HEAD_LINKS = ("head_yaw_link", "head_roll_link", "head_pitch_link")
TORSO_LINKS = {"torso_link", "waist_yaw_link", "waist_pitch_link", "waist_roll_link"}


def _num(s, n=3):
    return np.array([float(x) for x in s.split()]) if s else np.zeros(n)


def _quat(rpy):
    x, y, z, w = Rotation.from_euler("xyz", rpy).as_quat()  # URDF rpy = fixed-axis X, Y, Z
    return [w, x, y, z]


def build_spec(urdf=URDF, verbose=True):
    """URDF -> MjSpec: upper body, visual meshes only (no collisions); the torso and head are not rendered."""
    t0 = time.time()
    r = ET.parse(str(urdf)).getroot()
    links = {link.get("name"): link for link in r.findall("link")}
    joints = r.findall("joint")
    children = {}
    for j in joints:
        children.setdefault(j.find("parent").get("link"), []).append(j)
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.option.gravity = [0, 0, 0]
    meshdir = Path(urdf).parent
    # legs are never visible, the camera sits inside the head
    skip = {n for n in links if n.startswith(LEG_PREFIX) or n in HEAD_LINKS or n.startswith("IMU")}
    render_links = {n for n in links if n not in skip and n != "base_link"} - TORSO_LINKS
    nmesh = 0

    def add_link(parent_body, lname):
        nonlocal nmesh
        link = links[lname]
        body = parent_body
        if lname in render_links:
            for vi, vis in enumerate(link.findall("visual")):
                m = vis.find("geometry/mesh")
                if m is None:
                    continue  # tip marker spheres
                f = (meshdir / m.get("filename")).resolve()
                mname = f"m_{lname}_{vi}"
                spec.add_mesh(name=mname, file=str(f), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
                o = vis.find("origin")
                g = body.add_geom(name=f"g_{lname}_{vi}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname,
                                  pos=_num(o.get("xyz")) if o is not None else [0, 0, 0],
                                  quat=_quat(_num(o.get("rpy"))) if o is not None else [1, 0, 0, 0])
                g.contype, g.conaffinity, g.group, g.mass = 0, 0, 2, 0.01
                nmesh += 1
        for j in children.get(lname, []):
            c = j.find("child").get("link")
            if c in skip:
                continue
            o = j.find("origin")
            b = body.add_body(name=c, pos=_num(o.get("xyz")) if o is not None else [0, 0, 0],
                              quat=_quat(_num(o.get("rpy"))) if o is not None else [1, 0, 0, 0])
            b.add_site(name=f"s_{c}", pos=[0, 0, 0])
            if j.get("type") == "revolute":
                lim = j.find("limit")
                b.add_joint(name=j.get("name"), type=mujoco.mjtJoint.mjJNT_HINGE, axis=_num(j.find("axis").get("xyz")),
                            range=[float(lim.get("lower")), float(lim.get("upper"))],
                            limited=mujoco.mjtLimited.mjLIMITED_FALSE)
            b.mass, b.inertia, b.explicitinertial = 0.01, [1e-5, 1e-5, 1e-5], True
            add_link(b, c)

    base = spec.worldbody.add_body(name="base_link")
    base.mass, base.inertia, base.explicitinertial = 1.0, [1e-3, 1e-3, 1e-3], True
    add_link(base, "base_link")
    # the head links are skipped above but the camera hangs off head_pitch_link: rebuild just that chain, no geoms
    parent = spec.body("torso_link")
    for jn in NECK:
        j = next(x for x in joints if x.get("name") == jn)
        c = j.find("child").get("link")
        o = j.find("origin")
        b = parent.add_body(name=c, pos=_num(o.get("xyz")), quat=_quat(_num(o.get("rpy"))))
        b.add_joint(name=jn, type=mujoco.mjtJoint.mjJNT_HINGE, axis=_num(j.find("axis").get("xyz")))
        b.mass, b.inertia, b.explicitinertial = 0.01, [1e-5, 1e-5, 1e-5], True
        parent = b
    if verbose:
        print(f"[gr1] spec built: {nmesh} mesh geoms in {time.time() - t0:.1f}s", flush=True)
    return spec


class GR1Embodiment:
    name = "gr1_fourier"

    def __init__(self, verbose=True):
        t0 = time.time()
        self.model = m = build_spec(verbose=verbose).compile()
        self.data = mujoco.MjData(m)
        if verbose:
            print(f"[gr1] compiled: nbody={m.nbody} njnt={m.njnt} ngeom={m.ngeom} nmesh={m.nmesh} "
                  f"in {time.time() - t0:.1f}s", flush=True)

        def qa(n):
            return int(m.jnt_qposadr[m.joint(n).id])

        # (q44 index, qpos address, multiplier): arms, neck, waist, then the hands (6 values -> 11 joints, signs from
        # the URDF limits, mimic multipliers from the URDF)
        jmap = []
        for side, sl in (("left", SL["left"]), ("right", SL["right"])):
            for k, jn in enumerate(ARM_JOINTS):
                jmap.append((sl.start + k, qa(f"{side}_{jn}_joint"), 1.0))
        for k, jn in enumerate(NECK):
            jmap.append((SL["neck"].start + k, qa(jn), 1.0))
        for k, jn in enumerate(WAIST):
            jmap.append((SL["waist"].start + k, qa(jn), 1.0))
        for P, hs in (("L", SL["left_hand"]), ("R", SL["right_hand"])):
            for k, f in enumerate(FINGERS):
                jmap.append((hs.start + k, qa(f"{P}_{f}_proximal_joint"), -1.0))
                jmap.append((hs.start + k, qa(f"{P}_{f}_intermediate_joint"), -0.975))
            jmap.append((hs.start + 4, qa(f"{P}_thumb_proximal_pitch_joint"), 1.0))
            jmap.append((hs.start + 4, qa(f"{P}_thumb_distal_joint"), 1.06))
            jmap.append((hs.start + 5, qa(f"{P}_thumb_proximal_yaw_joint"), -1.0))
        self._src = np.array([a for a, _, _ in jmap])
        self._dst = np.array([b for _, b, _ in jmap])
        self._mul = np.array([c for _, _, c in jmap])
        self.geomgroup = np.array([0, 0, 1, 0, 0, 0], np.uint8)
        # parts: arm links / hand (palm + proximal phalanges) / pads (distal phalanges = intermediate + thumb distal)
        self.geom_part = np.full(m.ngeom, PART_ARM, np.int32)
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            if bn[:2] in ("L_", "R_"):
                self.geom_part[g] = PART_PAD if ("intermediate" in bn or "distal" in bn) else PART_GRIPPER
        self.q44 = np.zeros(44)
        mujoco.mj_kinematics(m, self.data)

    def set_state(self, q44, grip=None):
        """Pose the robot; `grip` is ignored (the hands are posed by their joints in q44)."""
        q44 = np.asarray(q44, float)
        self.q44 = q44
        qp = self.data.qpos
        qp[:] = 0
        qp[self._dst] = q44[self._src] * self._mul
        mujoco.mj_kinematics(self.model, self.data)

    def body_pose(self, name):
        b = self.model.body(name).id
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def head_pose(self):
        """4x4 world pose of head_pitch_link (FK through waist + neck)."""
        p, R = self.body_pose("head_pitch_link")
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T
