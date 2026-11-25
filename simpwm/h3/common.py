"""Shared pieces of the MiniMax-H3 Ref2VA LoRA (SimPWM-H3): geometry, prompt, data items, VAE / layout helpers.

Geometry (fixed for training and testing):
  target 704x1280 (real 720p frames stretched 720->704, as for LTX), the 97-step 15 fps clip resampled to H3's 24 fps
  grid: 158 frames (17*9+5; source frame floor(i*15/24), the last frames repeat step 96), 47 latent frames of 44x80
  (22x40 = 880 rows per latent frame).
References, in this order: <Video 1> = RGB proxy at 352x640 (half resolution, as LTX's IC-LoRA reference),
  <Picture 1> = real first frame at 704x1280.
First-frame pin: target latent frame 0 = single-frame VAE latent of the real first frame (704x1280, the same latent as
  Picture 1), kept at t=0.999 and excluded from the loss; the sampler never updates it.
Text: the follow prompt (PROMPT_HEAD_FOLLOW) + the dataset caption; Qwen3-VL sees the 352x640 proxy and the 704x1280
  Picture 1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from simpwm import paths

MODEL = paths.H3_MODEL  # MiniMax-H3 Ref2VA in diffusers layout (docs/installation.md)

H, W = 704, 1280                      # target
PH, PW = 352, 640                     # proxy reference
PIC_HW = (H, W)                       # Picture 1 and the pinned frame
SRC_FRAMES, SRC_FPS, FPS = 97, 15.0, 24
NUM_FRAMES = 158                      # 17*9+5
NUM_LATENT_FRAMES = 47                # 5*9+2
LAT_H, LAT_W = H // 16, W // 16
P_LAT_H, P_LAT_W = PH // 16, PW // 16
AUDIO_LATENTS = int(round(NUM_FRAMES / FPS * 40))     # 263 per channel
COND_T = 0.999
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

PROMPT_HEAD_FOLLOW = (
    "subject_definitions:\n"
    "<Video 1> is a simulator rendering of the robot arm and the scene, seen from the same fixed camera as the target video.\n"
    "<Picture 1> is the first frame of the target video, a real photograph from that camera.\n\n"
    "instruction:\n"
    "The target video re-renders <Video 1> as a photorealistic real-camera recording that starts from <Picture 1>. "
    "It follows <Video 1> frame by frame: the arm pose, the gripper opening and closing, and every object's position "
    "and contact. Objects move only when and as they move in <Video 1>; an object is lifted only if it is lifted in "
    "<Video 1>, and an object that stays on the table in <Video 1> stays on the table.\n\n"
    "detailed_description:\n"
)


def prompt_for(caption: str) -> str:
    return PROMPT_HEAD_FOLLOW + caption.strip()


def items(data, ids=None, target=True):
    """Samples of a paired dataset in the layout of video/dataset.py (DATA/dataset.jsonl rows {caption, video,
    reference_video, idx}; first frame DATA/first/<idx>.png, or the row's "first"): dicts name ("<idx>"), proxy, first,
    caption, target (the real video, None with target=False or without one). ids: optional file with the idx values to
    keep, in that order (ids without a row are reported and skipped)."""
    data = Path(data)
    rows = {str(r["idx"]): r for r in map(json.loads, open(data / "dataset.jsonl"))}
    keep = [str(int(i)) if i.isdigit() else i for i in open(ids).read().split()] if ids else list(rows)
    missing = [i for i in keep if i not in rows]
    if missing:  # episodes whose proxy failed are not in the dataset
        print(f"{len(missing)} of {len(keep)} ids are not in {data}/dataset.jsonl, skipped: {' '.join(missing[:20])}", flush=True)
        keep = [i for i in keep if i in rows]
    return [dict(name=i, proxy=data / rows[i]["reference_video"], first=data / rows[i].get("first", f"first/{i}.png"),
                 caption=rows[i]["caption"],
                 target=data / rows[i]["video"] if target and rows[i].get("video") else None) for i in keep]


def read_video(path, n=None) -> np.ndarray:
    import av
    out = []
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            out.append(f.to_ndarray(format="rgb24"))
            if n and len(out) >= n:
                break
    return np.stack(out)


def resize_frames(frames: np.ndarray, h=H, w=W) -> np.ndarray:
    import cv2
    if frames.shape[1:3] == (h, w):
        return frames
    interp = cv2.INTER_AREA if frames.shape[1] >= h else cv2.INTER_CUBIC
    return np.stack([cv2.resize(f, (w, h), interpolation=interp) for f in frames])


def to_24fps(frames97: np.ndarray) -> np.ndarray:
    """97 frames at 15 fps -> 158 frames at 24 fps (nearest earlier-or-equal slot, last frames held)."""
    idx = np.minimum(np.floor(np.arange(NUM_FRAMES) * SRC_FPS / FPS + 1e-6).astype(int), len(frames97) - 1)
    return frames97[idx]


def load_clip(path, hw=(H, W), n=SRC_FRAMES) -> np.ndarray:
    f = read_video(path, n)
    assert len(f) >= n, (path, len(f))
    return to_24fps(resize_frames(f[:n], *hw))


def load_image(path, hw=(H, W)) -> np.ndarray:
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB"))
    return resize_frames(im[None], *hw)[0]


@torch.no_grad()
def vae_encode(vae, frames_uint8: np.ndarray, device, seed=42) -> torch.Tensor:
    """(T,H,W,3) uint8 -> normalized latents (24, t, h, w) float32, the released conditioning recipe."""
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    px = torch.from_numpy(np.ascontiguousarray(frames_uint8)).to(device).permute(3, 0, 1, 2)[None]
    with torch.autocast("cuda", dtype=torch.float16):
        lat = encode_vae_condition(vae, px, PIXEL_MEAN, PIXEL_STD, seed)
    return lat[0].float()


def patchify(lat: torch.Tensor) -> torch.Tensor:
    """(24, T, h, w) -> (T*h/2*w/2, 96) rows, frame-major."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents
    return patchify_video_latents(lat[None], (1, 2, 2))


def unpatchify(rows: torch.Tensor, t=NUM_LATENT_FRAMES, h=LAT_H, w=LAT_W) -> torch.Tensor:
    rows = rows.reshape(t, h // 2, w // 2, 24, 1, 2, 2).permute(3, 0, 4, 1, 5, 2, 6)
    return rows.reshape(24, t, h, w)


class _Ref:
    def __init__(self, kind, has_audio=False):
        self.kind, self.has_audio = kind, has_audio


def build_layout(text_tags: torch.Tensor):
    """Packed [text | Video 1 (proxy, 352x640) | Picture 1 (704x1280) | target audio | target video] layout."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep
    refs = [_Ref("video"), _Ref("image")]
    cond = [torch.empty(1, 24, NUM_LATENT_FRAMES, P_LAT_H, P_LAT_W), torch.empty(1, 24, 1, PIC_HW[0] // 16, PIC_HW[1] // 16)]
    pos, tags, vidx, aidx, tidx, ncv, nca = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        text_tags, refs, cond, [], NUM_LATENT_FRAMES, LAT_H, LAT_W, AUDIO_LATENTS, (1, 2, 2), 2, 2, 0)
    assert nca == 0
    return dict(position_ids=pos, token_tags=tags, video_indices=vidx, audio_indices=aidx, text_indices=tidx,
                n_cond_video=ncv, rows_per_frame=(LAT_H // 2) * (LAT_W // 2))


def row_timesteps(layout, n_text, t_video: float, t_audio: float, pin_rows: int):
    vidx, aidx = layout["video_indices"], layout["audio_indices"]
    seq = layout["position_ids"].shape[0]
    t = torch.full((seq,), float(t_video), dtype=torch.float32)
    ncv = layout["n_cond_video"]
    t[vidx[:ncv]] = COND_T
    t[vidx[ncv:ncv + pin_rows]] = COND_T
    t[aidx] = float(t_audio)
    return torch.unique(t, sorted=True, return_inverse=True)


def shift_sigma(u, s):
    return s * u / (1 + (s - 1) * u)
