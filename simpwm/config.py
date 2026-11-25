"""Simulator settings.

One `SimConfig` instance, `config.cfg`, is read by the simulator modules. It starts from the preset of the dataset being
run (`use("droid")` / `use("droid_hires")`): the configuration reported in the paper. Ablations override single fields,
either in code (`config.update(snap=False)`) or from the environment, ``SIMPWM_<FIELD>=0|1`` (e.g. ``SIMPWM_SNAP=0``); an
override survives later `use()` calls. Every proxy run records the effective settings (`config.snapshot()`) in its
``info.json``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class SimConfig:
    # real-to-sim
    sam3_proposals: bool = True  # SAM 3 concept instances as object proposals (ablation: off)
    snap: bool = True  # action-conditioned depth snap: an object the gripper closes on sits at the gripper's depth
    pin: bool = True  # snapped objects stay where they were snapped until grasped (no gravity before the grasp)
    logged_camera: bool = False  # use the camera as logged by the dataset instead of the calibrated one (ablation)
    # gripper-signal rules (they need the *measured* gripper position, not a commanded one)
    empty_closures: bool = True  # a closure that closes fully held nothing: no proposal, snap or grasp there
    phantoms: bool = True  # a closure that stops short with no object between the pads adds a phantom object
    phantom_two_pass: bool = True  # ... only where a rollout of the initial state grasps nothing in that window
    # object behaviour
    soft_objects: bool = True  # VLM-named deformables hang from the jaws / slide as a sheet
    # high-resolution proxies (preset "droid_hires"; off in the configuration reported in the paper)
    hires: bool = False  # simulate at 640x360 instead of 320x180 (sim/scale.py); SAM 3, object masks, colours and
    # textures and the background depth of the full-size proxy render come from the full-resolution first frame
    jaw_inside: bool = False  # grasp / snap / phantom test on the jaw volume between the pads (engine.jaw_inside)
    parts: bool = False  # SAM 3 part prompts (lid, cap, knob, ...) so that a part becomes its own object
    parts_nested: bool = False  # ... kept only inside a generic object instance (a lid in a pot, not a cabinet door)
    carry_children: bool = False  # what rests on / in a grasped rigid object is carried with it


PRESETS = {
    "droid": SimConfig(),
    "droid_hires": SimConfig(hires=True, jaw_inside=True, parts=True, parts_nested=True, carry_children=True),
}

_overrides: dict = {}


def _env_overrides() -> dict:
    out = {}
    for f in fields(SimConfig):
        v = os.environ.get(f"SIMPWM_{f.name.upper()}")
        if v not in (None, ""):
            out[f.name] = v.lower() in ("1", "true", "yes", "on")
    return out


def use(dataset: str) -> SimConfig:
    """Switch to a dataset preset (overrides from `update` and the environment are kept)."""
    global cfg, dataset_name
    dataset_name = dataset
    cfg = replace(PRESETS[dataset], **{**_env_overrides(), **_overrides})
    return cfg


def update(**kw) -> SimConfig:
    """Override fields of the current configuration (kept across `use` calls)."""
    global cfg
    _overrides.update(kw)
    cfg = replace(cfg, **kw)
    return cfg


def snapshot() -> dict:
    return dict(dataset=dataset_name, **asdict(cfg))


dataset_name = "droid"
cfg = use("droid")
