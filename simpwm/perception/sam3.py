"""SAM 3 promptable concept segmentation: every instance of a short noun phrase in one call.

Weights: a Hugging Face format SAM 3 checkpoint, `simpwm.paths.SAM3_PATH` (repo id or local directory, env SAM3_PATH).
"""

from __future__ import annotations

import torch
from PIL import Image

from simpwm.paths import SAM3_PATH
from simpwm.perception import DEV, hf_model


@torch.no_grad()
def segment_concepts(img, phrases, threshold=0.4, mask_threshold=0.5):
    """img (H, W, 3) uint8; phrases list[str] -> list of dict(mask (H, W) bool, score, box, phrase)."""
    proc, model = hf_model("Sam3Processor", "Sam3Model", SAM3_PATH)
    out = []
    pil = Image.fromarray(img)
    for ph in phrases:
        inp = proc(images=pil, text=ph, return_tensors="pt").to(DEV)
        res = proc.post_process_instance_segmentation(model(**inp), threshold=threshold, mask_threshold=mask_threshold,
                                                      target_sizes=[img.shape[:2]])[0]
        for m, s, b in zip(res["masks"], res["scores"], res["boxes"]):
            out.append(dict(mask=m.cpu().numpy().astype(bool), score=float(s), box=b.cpu().tolist(), phrase=ph))
    return out
