"""Evaluation reference tracks from SAM 3's tracker (independent of the SAM 2 tracks used to build the proxy).

Same interface as tracks.track: frames (T, H, W, 3) uint8, masks0 {obj_id: (H, W) bool} -> {obj_id: (T, H, W) bool},
with the per-frame presence logits of the last call in track.scores.
"""

from __future__ import annotations

import torch

from simpwm.eval.tracks import propagate
from simpwm.perception import DEV
from simpwm.perception.sam3 import SAM3_PATH

_V = {}


def _video_model():
    if "v" not in _V:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        _V["v"] = (Sam3TrackerVideoProcessor.from_pretrained(SAM3_PATH),
                   Sam3TrackerVideoModel.from_pretrained(SAM3_PATH).to(DEV, torch.bfloat16).eval())
    return _V["v"]


def track(frames, masks0):
    out, track.scores = propagate(*_video_model(), frames, masks0)
    return out
