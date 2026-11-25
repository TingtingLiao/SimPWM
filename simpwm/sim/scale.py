"""Resolution of the DROID simulation frame and the pixel / point constants that depend on it.

The simulation works on the loader's frames (droid/data.py) with the camera intrinsics of that size: 320x180, the RLDS
image size, or 640x360 with `config.cfg.hires`. At the higher resolution every pixel-unit constant is scaled by
s = W / 320 (linear: distances, kernel radii) or by s^2 (pixel counts and point counts: the object point clouds have
one point per pixel). Metric constants (metres) never change. With `hires` off, s = 1 and every helper returns its
argument unchanged.
"""

from __future__ import annotations

from simpwm import config

BASE_HW = (180, 320)
HIRES_HW = (360, 640)


def sim_hw():
    """(H, W) of the simulation frame."""
    return HIRES_HW if config.cfg.hires else BASE_HW


def s():
    return sim_hw()[1] / 320.0


def px(n):
    """A linear pixel distance at 320x180 -> the same distance at the simulation resolution (int in, int out)."""
    return n if s() == 1 else (round(n * s()) if isinstance(n, int) else n * s())


def cnt(n):
    """A pixel or point count at 320x180 -> the same area / density at the simulation resolution."""
    return n if s() == 1 else (round(n * s() * s()) if isinstance(n, int) else n * s() * s())


def k(n):
    """An odd kernel size 2r+1 at 320x180 -> 2*round(r*s)+1."""
    return n if s() == 1 else 2 * round((n - 1) / 2 * s()) + 1
