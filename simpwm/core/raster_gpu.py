"""GPU robot rasterizer (nvdiffrast) with the outputs of Embodiment.render's MuJoCo ray caster, for every robot render
of the pipeline and for the inner loops that render the robot thousands of times (camera fitting).

The render geoms (Embodiment.geomgroup) are all meshes; their vertices are uploaded once in geom-local coordinates.
world_verts() poses them for a batch of robot states (MuJoCo kinematics on CPU, one small matmul per geom on GPU),
parts() projects them with a Camera, radial distortion included (applied per vertex: the meshes are fine enough that
straight triangle edges are a negligible error), and rasterizes at pixel centres, as the ray caster samples. Pixels
whose undistorted ray lies outside the distortion's invertible range are misses, as in Camera.valid_rays.

The states stay fixed while a camera is fitted, so callers pose the vertices once and only re-project per camera.

  gr = GpuRaster(emb)
  V = gr.world_verts([(q0, g0), (q1, g1)])      # (B, V, 3) on GPU
  part = gr.parts(V, cam)                        # (B, H, W) int16 torch, -1 = miss
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

NEAR, FAR = 0.01, 50.0


class GpuRaster:
    def __init__(self, emb, device="cuda"):
        import nvdiffrast.torch as dr

        self.emb, self.dev = emb, torch.device(device)
        self.ctx = dr.RasterizeCudaContext(device=self.dev)
        self._dr = dr
        m = emb.model
        geoms = [g for g in range(m.ngeom) if emb.geomgroup[m.geom_group[g]]]
        assert all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in geoms), "only mesh render geoms supported"
        verts, faces, gid, off = [], [], [], 0
        for i, g in enumerate(geoms):
            d = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[d]:m.mesh_vertadr[d] + m.mesh_vertnum[d]]
            f = m.mesh_face[m.mesh_faceadr[d]:m.mesh_faceadr[d] + m.mesh_facenum[d]]
            verts.append(v), faces.append(f + off), gid.append(np.full(len(v), i))
            off += len(v)
        self.geoms = np.array(geoms)
        self.v_local = torch.tensor(np.concatenate(verts), dtype=torch.float32, device=self.dev)
        self.v_geom = torch.tensor(np.concatenate(gid), dtype=torch.long, device=self.dev)
        self.faces = torch.tensor(np.concatenate(faces), dtype=torch.int32, device=self.dev)
        f_geom = self.v_geom[self.faces[:, 0].long()]
        # triangle id (1-based in nvdiffrast, 0 = empty) -> part id; index 0 -> -1
        part = torch.tensor(emb.geom_part[self.geoms], dtype=torch.int16, device=self.dev)
        self.tri_part = torch.cat([torch.tensor([-1], dtype=torch.int16, device=self.dev), part[f_geom]])

    def world_verts(self, states):
        """(B, V, 3) world vertices for robot states [(arm_q, grip), ...] (the embodiment's current gripper yaw)."""
        Rs, ts = [], []
        for q, g in states:
            self.emb.set_state(q, g)
            d = self.emb.data
            Rs.append(d.geom_xmat[self.geoms].reshape(-1, 3, 3).copy()), ts.append(d.geom_xpos[self.geoms].copy())
        R = torch.tensor(np.stack(Rs), dtype=torch.float32, device=self.dev)[:, self.v_geom]  # (B, V, 3, 3)
        t = torch.tensor(np.stack(ts), dtype=torch.float32, device=self.dev)[:, self.v_geom]  # (B, V, 3)
        return torch.einsum("bvij,vj->bvi", R, self.v_local) + t

    def clip(self, V, cam):
        """World vertices (B, V, 3) -> nvdiffrast clip coordinates (B, V, 4) for cam (pixel centres at +0.5)."""
        T_cw = torch.tensor(np.linalg.inv(cam.T_wc), dtype=torch.float32, device=self.dev)
        Xc = V @ T_cw[:3, :3].T + T_cw[:3, 3]
        z = Xc[..., 2:3]
        # distorted image-plane point times z (Xd = xd * z). Vertices behind / at the near plane or beyond the
        # distortion's invertible radius keep the plain pinhole Xd = (X, Y): clip coordinates stay linear in the
        # camera-frame vertex, so nvdiffrast's homogeneous clipping cuts their triangles correctly (moving such
        # vertices out of view instead lets triangles that cross the camera plane cover the whole image).
        Xd = Xc[..., :2]
        k1, k2 = cam.dist
        if (k1, k2) != (0.0, 0.0):
            front = z > NEAR
            xn = Xd / torch.where(front, z, torch.ones_like(z))
            r2 = (xn ** 2).sum(-1, keepdim=True)
            r_max = float(cam._radial_table()[0][-1])  # beyond it the model folds back; those pixels are misses
            Xd = torch.where(front & (r2 <= r_max ** 2), Xd * (1 + k1 * r2 + k2 * r2 * r2), Xd)
        K = cam.K
        U = Xd[..., 0:1] * K[0, 0] + K[0, 2] * z  # u * z
        V_ = Xd[..., 1:2] * K[1, 1] + K[1, 2] * z
        # nvdiffrast: row 0 at y_ndc = -1; z_ndc * z is the standard perspective depth row
        z_clip = (FAR + NEAR) / (FAR - NEAR) * z - 2 * FAR * NEAR / (FAR - NEAR)
        return torch.cat([2 * U / cam.W - z, 2 * V_ / cam.H - z, z_clip, z], -1)

    def parts(self, V, cam):
        """(B, H, W) int16 part ids (-1 = miss) of the posed vertices V seen from cam."""
        rast, _ = self._dr.rasterize(self.ctx, self.clip(V, cam).contiguous(), self.faces, resolution=[cam.H, cam.W])
        part = self.tri_part[rast[..., 3].long()]
        if cam.dist != (0.0, 0.0):
            part = torch.where(self._valid(cam), part, torch.full_like(part, -1))
        return part

    def render(self, cam, shade=False):
        """Embodiment.render for the embodiment's CURRENT state: z-depth (H, W) float32 (0 = miss), part id (H, W)
        int16 (-1 = miss), and with shade=True the Lambert-shaded RGB (H, W, 3) (flat face normals, turned to the
        camera, as the ray caster's hit normals)."""
        from simpwm.core.embodiment import shade_color

        d = self.emb.data
        R = torch.tensor(d.geom_xmat[self.geoms].reshape(-1, 3, 3), dtype=torch.float32, device=self.dev)[self.v_geom]
        t = torch.tensor(d.geom_xpos[self.geoms], dtype=torch.float32, device=self.dev)[self.v_geom]
        V = (torch.einsum("vij,vj->vi", R, self.v_local) + t)[None]
        clip = self.clip(V, cam).contiguous()
        rast, _ = self._dr.rasterize(self.ctx, clip, self.faces, resolution=[cam.H, cam.W])
        tri = rast[0, ..., 3].long()
        hit = tri > 0
        if cam.dist != (0.0, 0.0):
            hit &= self._valid(cam)
        z = self._dr.interpolate(clip[..., 3:4].contiguous(), rast, self.faces)[0][0, ..., 0]
        z = torch.where(hit, z, torch.zeros_like(z))
        part = torch.where(hit, self.tri_part[tri], torch.full_like(tri, -1, dtype=torch.int16))
        out = z.cpu().numpy().astype(np.float32), part.cpu().numpy().astype(np.int16)
        if not shade:
            return out
        f = self.faces.long()
        a, b, c = V[0, f[:, 0]], V[0, f[:, 1]], V[0, f[:, 2]]
        n = torch.nn.functional.normalize(torch.cross(b - a, c - a, dim=-1), dim=-1)
        eye = torch.tensor(cam.T_wc[:3, 3], dtype=torch.float32, device=self.dev)
        n = torch.where(((eye - a) * n).sum(-1, keepdim=True) < 0, -n, n)
        hi = hit.cpu().numpy()
        ti = (tri[hit] - 1)
        geomid = self.geoms[self.v_geom[f[ti, 0]].cpu().numpy()]
        col = np.zeros((cam.H, cam.W, 3), np.float32)
        col[hi] = shade_color(self.emb.model, geomid, n[ti].cpu().numpy().astype(float), cam)
        return out + (col,)

    @classmethod
    def of(cls, emb):
        """The embodiment's shared rasterizer, or None without CUDA / nvdiffrast (then the CPU ray caster is used)."""
        if not hasattr(emb, "_gpu_raster"):
            emb._gpu_raster = None
            if torch.cuda.is_available():
                try:
                    emb._gpu_raster = cls(emb)
                except ImportError:
                    pass
        return emb._gpu_raster

    def _valid(self, cam):
        # valid-ray mask per camera intrinsics; several are cached because the pipeline alternates the simulation
        # camera and the reference-resolution camera every step
        key = (cam.H, cam.W, cam.dist, tuple(cam.K.ravel()))
        cache = self.__dict__.setdefault("_valid_cache", {})
        if key not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[key] = torch.tensor(cam.valid_rays().reshape(cam.H, cam.W), device=self.dev)
        return cache[key]


class PosedRobot:
    """Robot silhouettes for a fixed list of states under changing cameras (camera fitting): on the GPU when available
    (the states are posed once), else Embodiment.render per state (the CPU ray caster; see its note on missed geoms)."""

    def __init__(self, emb, states):
        self.emb, self.states, self.gr = emb, list(states), GpuRaster.of(emb)
        self.V = self.gr.world_verts(self.states) if self.gr is not None and self.states else None

    def masks(self, cam):
        """(B, H, W) bool numpy robot masks."""
        if not self.states:
            return np.zeros((0, cam.H, cam.W), bool)
        if self.V is not None:
            return (self.gr.parts(self.V, cam) >= 0).cpu().numpy()
        out = []
        for q, g in self.states:
            self.emb.set_state(q, g)
            out.append(self.emb.render(cam)[1] >= 0)
        return np.stack(out)
