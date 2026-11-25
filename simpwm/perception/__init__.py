"""Off-the-shelf perception models used by the simulator: Grounding DINO open-vocabulary boxes, SAM2 masks and Depth
Anything V2 metric depth (SAM 3 concept segmentation is in `sam3`).

Models are Hugging Face checkpoints, loaded once on first use, on CUDA when available (else CPU).
"""

from __future__ import annotations

import torch
from PIL import Image

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_MODELS = {}


def hf_model(processor_cls: str, model_cls: str, repo: str):
    """(processor, model) of a Hugging Face checkpoint (transformers class names), loaded once and kept."""
    key = (model_cls, repo)
    if key not in _MODELS:
        import transformers

        proc = getattr(transformers, processor_cls).from_pretrained(repo)
        _MODELS[key] = (proc, getattr(transformers, model_cls).from_pretrained(repo).to(DEV).eval())
    return _MODELS[key]


@torch.no_grad()
def detect(img, phrases, box_thr=0.25, text_thr=0.2):
    """Grounding DINO. img (H, W, 3) uint8, phrases list[str] -> list of dict(box=[x0, y0, x1, y1], score, label)."""
    proc, model = hf_model("AutoProcessor", "AutoModelForZeroShotObjectDetection", "IDEA-Research/grounding-dino-base")
    text = ". ".join(p.lower().strip(". ") for p in phrases) + "."
    inp = proc(images=Image.fromarray(img), text=text, return_tensors="pt").to(DEV)
    out = model(**inp)
    res = proc.post_process_grounded_object_detection(out, inp.input_ids, threshold=box_thr, text_threshold=text_thr,
                                                      target_sizes=[img.shape[:2]])[0]
    labels = res.get("text_labels", res.get("labels"))
    return [dict(box=b.tolist(), score=float(s), label=str(lab))
            for b, s, lab in zip(res["boxes"].cpu(), res["scores"].cpu(), labels)]


@torch.no_grad()
def segment(img, boxes=None, points=None, labels=None):
    """SAM2 image masks. boxes: list of [x0, y0, x1, y1] (one mask each), or points (N, 2) + labels (N,) for one mask.
    Returns (n, H, W) bool masks."""
    proc, model = hf_model("Sam2Processor", "Sam2Model", "facebook/sam2.1-hiera-large")
    kw = {}
    if boxes is not None:
        kw["input_boxes"] = [[list(map(float, b)) for b in boxes]]
    if points is not None:
        kw["input_points"] = [[[list(map(float, p)) for p in points]]]
        kw["input_labels"] = [[list(map(int, labels))]]
    inp = proc(images=Image.fromarray(img), return_tensors="pt", **kw).to(DEV)
    out = model(**inp, multimask_output=False)
    masks = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]
    return masks[:, 0].numpy().astype(bool)


@torch.no_grad()
def metric_depth(img):
    """Depth Anything V2 (metric, indoor): z-depth in metres (H, W)."""
    proc, model = hf_model("AutoImageProcessor", "AutoModelForDepthEstimation",
                           "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    inp = proc(images=Image.fromarray(img), return_tensors="pt").to(DEV)
    pred = model(**inp).predicted_depth
    pred = torch.nn.functional.interpolate(pred[:, None], size=img.shape[:2], mode="bicubic", align_corners=False)
    return pred[0, 0].float().cpu().numpy()
