"""NVlabs FoundationStereo (`paths.DEPS`/FoundationStereo, ViT-L checkpoint 23-51-11): disparity of a rectified stereo
pair, for droid/stereo/extract.py's fs stage. Runs in the FoundationStereo environment, with the checkout (and an empty
`open3d` stub package, which its utilities import but the depth path never uses) on PYTHONPATH; see docs/dataset.md.
"""

import torch
from omegaconf import OmegaConf

from simpwm import paths

FS = paths.DEPS / "FoundationStereo"
CKPT = FS / "pretrained_models/23-51-11"
REPO = "shriarul5273/FoundationStereo_models"  # mirror of the official Google Drive release


def load():
    if not (CKPT / "model_best_bp2.pth").exists():
        from huggingface_hub import hf_hub_download
        for f in ("cfg.yaml", "model_best_bp2.pth"):
            hf_hub_download(REPO, f"pretrained_models/23-51-11/{f}", local_dir=FS)
    from core.foundation_stereo import FoundationStereo
    cfg = OmegaConf.load(CKPT / "cfg.yaml")
    cfg.setdefault("vit_size", "vitl")
    cfg.valid_iters, cfg.hiera = 32, 0
    model = FoundationStereo(cfg)
    model.load_state_dict(torch.load(CKPT / "model_best_bp2.pth", map_location="cpu", weights_only=False)["model"])
    return model.cuda().eval(), cfg


def disparity(model, cfg, left, right):
    from core.utils.utils import InputPadder
    a = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    b = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    pad = InputPadder(a.shape, divis_by=32, force_square=False)
    a, b = pad.pad(a, b)
    with torch.no_grad(), torch.autocast("cuda"):
        d = model.forward(a, b, iters=cfg.valid_iters, test_mode=True)
    return pad.unpad(d.float()).cpu().numpy().reshape(left.shape[:2])
