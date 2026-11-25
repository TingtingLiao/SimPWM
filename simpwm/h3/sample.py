"""Sampling with the FSDP-sharded H3 Ref2VA model (each GPU its own video), optional LoRA, optional first-frame pin.

    torchrun --nproc-per-node 8 -m simpwm.h3.sample --cache CACHE --out DIR --names a,b|@file
                                                    --variant NAME:LORA|none:pin|nopin [--variant ...]
Reads the encodings of preencode.py; writes DIR/<variant>/<name>.mp4 (158 frames, 24 fps) and <name>_15fps.mp4
(97 frames, the 15 fps clip grid). Default 50-point schedule (49 forwards), video shift 12, audio shift 3, seed 0 per
sample.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file

from simpwm.h3 import model as h3model
from simpwm.h3.common import COND_T, MODEL, PIXEL_MEAN, PIXEL_STD, build_layout, patchify, row_timesteps, unpatchify


def write_mp4(path, frames, fps):
    import imageio
    imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)


@torch.no_grad()
def generate(model, cache, name, dev, steps, pin, seed=0):
    from diffusers import MiniMaxH3Scheduler
    v = load_file(str(cache / f"{name}.vae.safetensors"))
    t = load_file(str(cache / f"{name}.text.safetensors"))
    silence = load_file(str(cache / "silence_audio.safetensors"))["audio"]
    lay = build_layout(t["tags"])
    rpf, ncv = lay["rows_per_frame"], lay["n_cond_video"]
    g = torch.Generator().manual_seed(seed)
    cond = torch.cat([patchify(v["proxy"]), patchify(v["first"])])
    cond = COND_T * cond + (1 - COND_T) * torch.randn(cond.shape, generator=g)
    pin_rows = patchify(v["first"])
    x = torch.randn(lay["video_indices"].numel() - ncv, 96, generator=g)
    p = rpf if pin else 0
    if pin:
        x[:rpf] = COND_T * pin_rows + (1 - COND_T) * torch.randn(pin_rows.shape, generator=g)
    a = torch.randn(silence.shape, generator=g)
    x, a, cond = x.to(dev), a.to(dev), cond.to(dev)
    vs, as_ = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    vs.set_timesteps(steps, device=dev)
    as_.set_timesteps(steps, device=dev)
    kw = dict(encoder_hidden_states=t["hidden"].to(dev)[None], token_tags=lay["token_tags"].to(dev),
              position_ids=lay["position_ids"].to(dev), video_indices=lay["video_indices"].to(dev),
              audio_indices=lay["audio_indices"].to(dev), text_indices=lay["text_indices"].to(dev), return_dict=False)
    for i, (tv, ta) in enumerate(zip(vs.timesteps, as_.timesteps)):
        uniq, inv = row_timesteps(lay, t["hidden"].shape[0], float(tv), float(ta), p)
        pv, pa = model(hidden_states=torch.cat([cond, x])[None], audio_hidden_states=a[None], timestep=uniq.to(dev),
                       timestep_indices=inv.to(dev), **kw)
        x[p:] = vs.step(pv[0, ncv + p:].float(), tv, x[p:], return_dict=False)[0]
        a = as_.step(pa[0].float(), ta, a, return_dict=False)[0]
    if pin:
        x[:rpf] = pin_rows.to(dev)
    return unpatchify(x.float().cpu())


@torch.no_grad()
def decode(vae, lat, dev):
    m = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    s = torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    with torch.autocast("cuda", dtype=torch.float16):
        vid = vae.decode(lat[None].to(dev) * s + m, return_dict=False)[0]
    pm = torch.tensor(PIXEL_MEAN, device=dev).view(1, -1, 1, 1, 1)
    ps = torch.tensor(PIXEL_STD, device=dev).view(1, -1, 1, 1, 1)
    vid = (vid.float() * ps + pm).clamp(0, 1)[0]                       # (3,T,H,W)
    return (vid.permute(1, 2, 3, 0).cpu().numpy() * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="preencode.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--names", required=True, help="comma list or @file")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--rank", type=int, default=32)
    a = ap.parse_args()
    cache = Path(a.cache)
    rank, world, local, mesh = h3model.init_dist()
    dev = torch.device("cuda", local)
    log = lambda *m: print(f"[{time.strftime('%H:%M:%S')}] r{rank}", *m, flush=True)
    model, mods = h3model.build(str(MODEL), mesh, lora_rank=a.rank, log=log if rank == 0 else (lambda *m: None))
    model.eval()
    from diffusers import AutoencoderKLMiniMaxH3
    vae = AutoencoderKLMiniMaxH3.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float16).to(dev).eval()
    names = (open(a.names[1:]).read() if a.names.startswith("@") else a.names).replace(",", " ").split()
    for var in a.variant:
        vname, lora, pin = var.split(":")
        if lora == "none":
            h3model.set_lora(mods, False)
        else:
            h3model.set_lora(mods, True)
            h3model.load_lora(mods, lora)
        od = os.path.join(a.out, vname)
        os.makedirs(od, exist_ok=True)
        todo = [n for n in names if not os.path.exists(os.path.join(od, f"{n}.mp4"))]
        dist.barrier()
        if not todo:
            continue
        nreal = len(todo)
        while len(todo) % world:
            todo.append(todo[-1])                                      # every rank must run the same number of forwards
        for k in range(rank, len(todo), world):
            nm = todo[k]
            t0 = time.time()
            lat = generate(model, cache, nm, dev, a.steps, pin == "pin")
            tg = time.time() - t0
            frames = decode(vae, lat, dev)
            if k < nreal:
                write_mp4(os.path.join(od, f"{nm}.mp4"), frames, 24)
                i15 = np.minimum(np.ceil(np.arange(97) * 24 / 15 - 1e-9).astype(int), len(frames) - 1)  # step k = frame ceil(k * 24 / 15)
                write_mp4(os.path.join(od, f"{nm}_15fps.mp4"), frames[i15], 15)
            log(f"{vname} {nm} gen={tg:.0f}s total={time.time() - t0:.0f}s peak={torch.cuda.max_memory_allocated() / 2**30:.1f}G")
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
