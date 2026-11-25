"""Pinhole camera in the OpenCV convention (x right, y down, z forward) with a camera-to-world pose and optional radial
distortion (k1, k2) on normalized coordinates, as needed for wide-angle lenses such as the ZED 2 (~100 deg).

Pixel centres are at integer + 0.5 everywhere (`project`, `unproject`, `rays`).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class Camera:
    def __init__(self, K, T_wc, H, W, dist=(0.0, 0.0)):
        self.K = np.asarray(K, float)
        self.T_wc = np.asarray(T_wc, float)  # 4x4 camera -> world
        self.H, self.W = H, W
        self.dist = tuple(float(d) for d in dist)
        self._rays = None
        self._rt = None

    @classmethod
    def from_xyz_euler(cls, xyz_euler, K, H, W, euler="xyz"):
        """Camera from a pose given as [x, y, z, Euler angles (rad)], DROID's logged extrinsics format."""
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler(euler, xyz_euler[3:]).as_matrix()
        T[:3, 3] = xyz_euler[:3]
        return cls(K, T, H, W)

    def scaled(self, H, W):
        """The same camera at another image resolution."""
        s = np.diag([W / self.W, H / self.H, 1.0])
        return Camera(s @ self.K, self.T_wc, H, W, self.dist)

    # ---- radial distortion ----
    def _distort(self, xn):
        k1, k2 = self.dist
        r2 = (xn ** 2).sum(-1, keepdims=True)
        return xn * (1 + k1 * r2 + k2 * r2 * r2)

    def _radial_table(self, r_max=4.0, n=40001):
        """r -> r_d = r (1 + k1 r^2 + k2 r^4) on its monotonic range (from 0 up to the first turning point)."""
        if self._rt is None:
            k1, k2 = self.dist
            r = np.linspace(0.0, r_max, n)
            rd = r * (1 + k1 * r ** 2 + k2 * r ** 4)
            stop = np.nonzero(np.diff(rd) <= 0)[0]
            m = stop[0] + 1 if len(stop) else n
            self._rt = (r[:m], rd[:m])
        return self._rt

    def _undistort(self, xd):
        """Inverse of _distort by interpolation on the monotonic radial table (a fixed-point iteration diverges for
        strong distortion at the image border). Beyond the invertible range the radius is clamped; `valid_rays`
        marks those pixels so renderers can treat them as misses."""
        if self.dist == (0.0, 0.0):
            return xd
        r_tab, rd_tab = self._radial_table()
        rd = np.linalg.norm(xd, axis=-1, keepdims=True)
        r = np.interp(rd, rd_tab, r_tab)
        return xd * np.where(rd > 1e-12, r / np.maximum(rd, 1e-12), 1.0)

    # ---- pixels <-> rays ----
    def _pixel_centres(self):
        return np.meshgrid(np.arange(self.W) + 0.5, np.arange(self.H) + 0.5)

    def _distorted_coords(self, u, v):
        """Pixels -> distorted normalized image coordinates (..., 2)."""
        return (np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(self.K).T)[..., :2]

    def _pix_dirs(self, u, v):
        """Pixels -> undistorted normalized rays (..., 3) with z = 1."""
        xn = self._undistort(self._distorted_coords(u, v))
        return np.concatenate([xn, np.ones_like(xn[..., :1])], -1)

    def valid_rays(self):
        """(H*W,) bool: the pixel's ray is inside the invertible range of the distortion model."""
        if self.dist == (0.0, 0.0):
            return np.ones(self.H * self.W, bool)
        xd = self._distorted_coords(*self._pixel_centres())
        return (np.linalg.norm(xd, axis=-1) <= self._radial_table()[1][-1]).reshape(-1)

    def rays(self):
        """World-frame ray origin (3,) and unit ray directions (H*W, 3), cached. Also sets `ray_z` (H*W,), the factor
        that converts a distance along a ray to z-depth."""
        if self._rays is None:
            d = self._pix_dirs(*self._pixel_centres()).reshape(-1, 3)
            n = np.linalg.norm(d, axis=1, keepdims=True)
            self.ray_z = (1.0 / n[:, 0])
            self._rays = (self.T_wc[:3, 3].copy(), np.ascontiguousarray((d / n) @ self.T_wc[:3, :3].T))
        return self._rays

    def project(self, X):
        """World points (N, 3) -> pixels (N, 2) and z (N,)."""
        T_cw = np.linalg.inv(self.T_wc)
        Xc = X @ T_cw[:3, :3].T + T_cw[:3, 3]
        xn = Xc[:, :2] / Xc[:, 2:3]
        xd = self._distort(xn) if self.dist != (0.0, 0.0) else xn
        uv = xd @ self.K[:2, :2].T + self.K[:2, 2]
        return uv, Xc[:, 2]

    def unproject(self, depth):
        """z-depth map (H, W) -> world points (H, W, 3)."""
        Xc = self._pix_dirs(*self._pixel_centres()) * depth[..., None]
        return Xc @ self.T_wc[:3, :3].T + self.T_wc[:3, 3]
