"""Embodiment: a kinematic robot model (MuJoCo) loaded from a per-robot JSON config.

The generic pipeline only uses this interface:
  emb.set_state(arm_q, grip)        # one logged robot state: joint positions, gripper closure in [0, 1]
  emb.render(cam) -> depth, part    # z-depth (0 = no hit) and per-pixel part id (-1 = background)
  emb.keypoints() -> {name: xyz}    # effector keypoints in the world (= robot base) frame
  emb.grip                          # normalized closure in [0, 1]

Everything robot specific (MJCF files from MuJoCo Menagerie under `simpwm.paths.ASSETS`, joint names, gripper mount,
effector keypoints) lives in resources/embodiments/<name>.json.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np

from simpwm.paths import ASSETS, RESOURCES

CONFIGS = RESOURCES / "embodiments"

# part ids written into the part map; the rasterizer maps them to semantic / instance ids
PART_ARM, PART_GRIPPER, PART_PAD = 0, 1, 2

LIGHT = np.array([-0.3, -0.5, -0.8])  # shading light direction, camera frame (towards the scene)


def light_factor(normal, cam):
    """Ambient + diffuse + facing term (n,) of the shading; `normal` (n, 3) are world-frame hit normals."""
    nc = normal @ cam.T_wc[:3, :3]  # world -> camera frame
    lam = np.clip(nc @ (-LIGHT / np.linalg.norm(LIGHT)), 0, 1)
    return 0.35 + 0.45 * lam + 0.3 * np.abs(nc[:, 2])


def shade_color(model, geomid, normal, cam, albedo=None):
    """Lambert-shaded colour (n rays, 3) in [0, 1] of the ray hits: `albedo` (n, 3) if given, else the geom's MJCF
    colour (material or geom rgba), times light_factor."""
    if albedo is None:
        g = np.maximum(geomid, 0)
        albedo = model.geom_rgba[g]
        if model.nmat:  # models without materials only have the geom colour
            mat = model.geom_matid[g]
            albedo = np.where((mat >= 0)[:, None], model.mat_rgba[np.maximum(mat, 0)], albedo)
        albedo = albedo[:, :3]
    return np.clip(albedo * light_factor(normal, cam)[:, None], 0, 1)


def _quat_z(deg):
    a = np.deg2rad(deg) / 2
    return [np.cos(a), 0.0, 0.0, np.sin(a)]


class Embodiment:
    def __init__(self, name: str):
        cfg = json.loads((CONFIGS / f"{name}.json").read_text())
        self.cfg = cfg
        spec = mujoco.MjSpec.from_file(str(ASSETS / cfg["arm_mjcf"]))
        self._grip_lut = None
        if cfg.get("gripper_mjcf"):
            gspec = mujoco.MjSpec.from_file(str(ASSETS / cfg["gripper_mjcf"]))
            self._grip_lut = self._gripper_lut(ASSETS / cfg["gripper_mjcf"], cfg)
            site = spec.site(cfg["attach_site"])
            site.quat = _quat_z(cfg.get("gripper_yaw_deg", 0.0))
            spec.attach(gspec, site=site, prefix="g_")
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in cfg["arm_joints"]])
        self.grip_qadr = None
        if self._grip_lut is not None:
            names, _ = self._grip_lut
            self.grip_qadr = np.array([m.jnt_qposadr[m.joint("g_" + j).id] for j in names])
        # geoms used for rendering: visual group only (collision meshes are coarse duplicates)
        vis = set(cfg.get("render_groups", [2]))
        self.geomgroup = np.array([1 if g in vis else 0 for g in range(6)], np.uint8)
        pad_bodies = {m.body(b).id for b in cfg.get("pad_bodies", [])}
        grip_root = m.body(cfg["gripper_root_body"]).id if cfg.get("gripper_root_body") else -1
        self.geom_part = np.full(m.ngeom, PART_ARM, np.int32)
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            chain = []
            while b > 0:
                chain.append(b)
                b = m.body_parentid[b]
            if pad_bodies & set(chain):
                self.geom_part[g] = PART_PAD
            elif grip_root in chain:
                self.geom_part[g] = PART_GRIPPER
        self.grip = 0.0
        if grip_root >= 0:
            self._mount_q0 = m.body_quat[grip_root].copy()
            self._yaw0 = self.yaw = cfg.get("gripper_yaw_deg", 0.0)

    def set_gripper_yaw(self, deg):
        """Rotate the gripper about the flange axis (mount yaw differs between labs/adapters)."""
        b = self.model.body(self.cfg["gripper_root_body"]).id
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array(_quat_z(deg - self._yaw0)), self._mount_q0)
        self.model.body_quat[b] = q
        self.yaw = deg

    @staticmethod
    def _gripper_lut(path, cfg, n=33):
        """The gripper linkage is a closed chain: tabulate its joint positions against closure (n steps) by settling
        the physics once per step."""
        m = mujoco.MjModel.from_xml_path(str(path))
        m.opt.gravity[:] = 0
        d = mujoco.MjData(m)
        names = [m.joint(j).name for j in range(m.njnt)]
        lo, hi = cfg["gripper_ctrl_range"]
        table = []
        for c in np.linspace(0, 1, n):
            d.ctrl[:] = lo + c * (hi - lo)
            for _ in range(1500):
                mujoco.mj_step(m, d)
            table.append(d.qpos[[m.jnt_qposadr[j] for j in range(m.njnt)]].copy())
        return names, np.array(table)

    def set_state(self, arm_q, grip=0.0):
        """Pose the model: arm joint positions and gripper closure in [0, 1] (interpolated in the linkage table)."""
        q = self.data.qpos
        q[self.arm_qadr] = np.asarray(arm_q, float)
        self.grip = float(np.clip(grip, 0, 1))
        if self._grip_lut is not None:
            _, tab = self._grip_lut
            x = self.grip * (len(tab) - 1)
            i = min(int(x), len(tab) - 2)
            w = x - i
            q[self.grip_qadr] = (1 - w) * tab[i] + w * tab[i + 1]
        mujoco.mj_kinematics(self.model, self.data)

    def body_pose(self, name):
        b = self.model.body(name).id
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def site_pose(self, name):
        s = self.model.site(name).id
        return self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3).copy()

    def keypoints(self):
        """{name: world position} of the effector keypoints of the config (a body or site plus an optional offset)."""
        out = {}
        for k, spec in self.cfg["keypoints"].items():
            p, R = self.body_pose(spec["body"]) if "body" in spec else self.site_pose(spec["site"])
            out[k] = p + R @ np.asarray(spec.get("offset", [0, 0, 0]), float)
        return out

    def render(self, cam, shade=False):
        """Render the robot from `cam`: z-depth (H, W) with 0 = miss, part id (H, W) with -1 = miss, and with
        shade=True also the Lambert-shaded RGB (H, W, 3) of the robot mesh (0 on a miss).

        With CUDA and nvdiffrast available the meshes are rasterized on the GPU (core/raster_gpu.py); otherwise they are
        ray-cast with MuJoCo's mj_multiRay on the CPU (no GL context needed). The ray caster misses a few mesh geoms
        (the FR3 link7_1 flange ring, ~2% of the robot's pixels) that the rasterizer hits."""
        from simpwm.core.raster_gpu import GpuRaster

        gr = GpuRaster.of(self)
        if gr is not None:
            return gr.render(cam, shade)
        origin, dirs = cam.rays()
        n = dirs.shape[0]
        geomid = np.full(n, -1, np.int32)
        dist = np.zeros(n)
        nrm = np.zeros(n * 3) if shade else None
        mujoco.mj_multiRay(self.model, self.data, origin, dirs.ravel(), self.geomgroup, 1, -1, geomid, dist, nrm, n,
                           50.0)
        hit = (geomid >= 0) & cam.valid_rays()
        z = np.where(hit, dist * cam.ray_z, 0.0)
        part = np.where(hit, self.geom_part[np.maximum(geomid, 0)], -1)
        out = z.reshape(cam.H, cam.W).astype(np.float32), part.reshape(cam.H, cam.W).astype(np.int16)
        if not shade:
            return out
        col = np.where(hit[:, None], shade_color(self.model, geomid, nrm.reshape(n, 3), cam), 0.0)
        return out + (col.reshape(cam.H, cam.W, 3).astype(np.float32),)
