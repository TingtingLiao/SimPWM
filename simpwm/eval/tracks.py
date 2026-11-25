"""Proxy runs against SAM 2 tracks of the real video (the per-run metrics of droid/run.py).

  track    SAM 2.1 video tracking of first-frame masks (also the tracker of the grasp-target extraction, grasp_iou.py)
  compare  robot IoU between the rendered robot and its track, per frame; object IoU and centroid error (px) between
           the proxy object masks and their tracks, next to the "object never moves" baseline (the first-frame mask
           held fixed)
"""

from __future__ import annotations

import numpy as np
import torch

from simpwm.eval.common import centroid
from simpwm.perception import DEV
from simpwm.sim import render as raster

_V = {}


def _video_model():
    if "v" not in _V:
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        rid = "facebook/sam2.1-hiera-large"
        _V["v"] = (Sam2VideoProcessor.from_pretrained(rid),
                   Sam2VideoModel.from_pretrained(rid).to(DEV, torch.bfloat16).eval())
    return _V["v"]


@torch.no_grad()
def propagate(proc, model, frames, masks0, keep_unknown=False):
    """Track the frame-0 masks with a transformers SAM 2 / SAM 3 video tracker -> ({obj_id: (T, H, W) bool},
    {obj_id: (T,) presence logits}). Object ids the tracker reports but was not given are kept only with
    keep_unknown."""
    T, H, W = frames.shape[:3]
    ids = [int(i) for i in masks0]
    masks0 = {int(k): v for k, v in masks0.items()}
    # frames and per-frame memories live on the CPU: long episodes otherwise take 50+ GB of GPU memory (OOM with
    # several proxy processes per GPU); only the storage moves, the computation is unchanged
    sess = proc.init_video_session(video=list(frames), inference_device=DEV, dtype=torch.bfloat16,
                                   video_storage_device="cpu", inference_state_device="cpu", processing_device="cpu")
    proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=0, obj_ids=list(ids),  # (consumes the list)
                                         input_masks=[masks0[i].astype(np.float32) for i in ids])
    model(inference_session=sess, frame_idx=0)  # condition on the first-frame masks before propagating
    out = {i: np.zeros((T, H, W), bool) for i in ids}
    scores = {i: np.full(T, np.nan, np.float32) for i in ids}
    for o in model.propagate_in_video_iterator(sess, start_frame_idx=0):
        m = proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
        for j, i in enumerate(o.object_ids):
            i = int(i)
            if i not in out:
                if not keep_unknown:
                    continue
                print(f"  track: unexpected object id {i!r} (known {ids})", flush=True)
                out[i] = np.zeros((T, H, W), bool)
            out[i][o.frame_idx] = m[j, 0].cpu().numpy().astype(bool)
            if o.object_score_logits is not None and i in scores:
                scores[i][o.frame_idx] = float(o.object_score_logits.reshape(-1)[j].float().cpu())
    return out, scores


def track(frames, masks0):
    """frames (T, H, W, 3) uint8; masks0 {obj_id: (H, W) bool} on frame 0 -> {obj_id: (T, H, W) bool}. The per-object
    presence logits of the last call (< 0: SAM 2 thinks the object is gone) are left in track.scores."""
    out, track.scores = propagate(*_video_model(), frames, masks0, keep_unknown=True)
    return out


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def compare(proxy_frames, tracked, robot_id="robot"):
    """proxy_frames: list of (depth, cat, contact). tracked: {id: (T,H,W)} incl. robot under `robot_id`."""
    T = len(proxy_frames)
    res = {}
    for oid, tr in tracked.items():
        ious, base, cerr, cerr_base = [], [], [], []
        for t in range(T):
            cat = proxy_frames[t][1]
            pm = raster.is_robot(cat) if oid == robot_id else cat == raster.C_OBJ0 + oid
            ious.append(_iou(pm, tr[t]))
            if oid != robot_id:
                base.append(_iou(tr[0], tr[t]))
                c1, c2, c0 = centroid(pm), centroid(tr[t]), centroid(tr[0])
                if c1 is not None and c2 is not None:
                    cerr.append(float(np.linalg.norm(c1 - c2)))
                if c0 is not None and c2 is not None:
                    cerr_base.append(float(np.linalg.norm(c0 - c2)))
        # motion: how far the tracked object actually moved (px), to separate moved vs. static objects
        cs = [centroid(tr[t]) for t in range(T)]
        cs = [c for c in cs if c is not None]
        moved = float(max(np.linalg.norm(c - cs[0]) for c in cs)) if cs else 0.0
        r = dict(iou=float(np.nanmean(ious)), iou_last=float(ious[-1]), iou_curve=[round(x, 3) for x in ious])
        if oid != robot_id:
            r.update(iou_static=float(np.nanmean(base)), iou_static_last=float(base[-1]),
                     cen_err_px=float(np.mean(cerr)) if cerr else None, cen_err_last_px=cerr[-1] if cerr else None,
                     cen_err_static_px=float(np.mean(cerr_base)) if cerr_base else None,
                     cen_err_static_last_px=cerr_base[-1] if cerr_base else None, moved_px=moved)
        res[str(oid)] = r
    return res
