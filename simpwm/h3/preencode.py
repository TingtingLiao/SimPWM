"""Offline encoding for the H3 LoRA: VAE latents (stage vae) and Qwen3-VL conditioning (stage text).

    python -m simpwm.h3.preencode --stage vae|text --data DS --cache CACHE [--ids FILE] [--no-target]
                                  [--shard i --nshards n] [--names a,b|@file]
Writes CACHE/<idx>.vae.safetensors  {target [24,47,44,80] (unless --no-target), proxy [24,47,22,40], first [24,1,44,80]}
       CACHE/<idx>.text.safetensors {hidden [L,5120] bf16, tags [L]}
       CACHE/silence_audio.safetensors {audio [526,32]} (normalized, channel-major rows)
DS is a paired dataset of video/dataset.py (--link layout: refs/ = the RGB proxy at 1280x704).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from simpwm.h3.common import (
    AUDIO_LATENTS,
    MODEL,
    NUM_FRAMES,
    PH,
    PIC_HW,
    PW,
    items,
    load_clip,
    load_image,
    prompt_for,
    vae_encode,
)


def atomic_save(path, tensors, meta=None):
    tmp = f"{path}.{os.getpid()}.tmp"
    save_file({k: v.contiguous() for k, v in tensors.items()}, tmp, metadata=meta or {})
    os.replace(tmp, path)


def stage_vae(todo, cache, dev):
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
    vae = AutoencoderKLMiniMaxH3.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float16).to(dev).eval()
    sil = cache / "silence_audio.safetensors"
    if not sil.exists():
        avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(MODEL, subfolder="audio_vae", torch_dtype=torch.float32).to(dev).eval()
        n = int(round(NUM_FRAMES / 24 * avae.config.sampling_rate))
        with torch.no_grad():
            post = avae.encode(torch.zeros(2, 1, n, device=dev), return_dict=False)[0]
        lat = post.mode().float().cpu().transpose(1, 2)            # (2, T, 32)
        mean = torch.tensor(avae.config.latents_mean).view(1, 1, -1)
        std = torch.tensor(avae.config.latents_std).view(1, 1, -1)
        rows = ((lat - mean) / std)[:, :AUDIO_LATENTS].reshape(-1, lat.shape[-1])
        print("silence audio rows", tuple(lat.shape), "->", tuple(rows.shape), flush=True)
        atomic_save(sil, {"audio": rows})
        del avae
    for it in todo:
        p = cache / f"{it['name']}.vae.safetensors"
        if p.exists():
            continue
        t0 = time.time()
        t = {"proxy": vae_encode(vae, load_clip(it["proxy"], (PH, PW)), dev).cpu(),
             "first": vae_encode(vae, load_image(it["first"], PIC_HW)[None], dev).cpu()}
        if it["target"] is not None:
            t["target"] = vae_encode(vae, load_clip(it["target"]), dev).cpu()
        atomic_save(p, t)
        print(it["name"], {k: tuple(v.shape) for k, v in t.items()}, f"{time.time() - t0:.1f}s",
              f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)


def stage_text(todo, cache, dev):
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3Ref2VATextEncoderStep,
        get_qwen3vl_prompt_embeds,
    )
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference, MiniMaxH3VideoReference
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    te = Qwen3VLForConditionalGeneration.from_pretrained(MODEL / "text_encoder", torch_dtype=torch.bfloat16).to(dev).eval()
    proc = Qwen3VLProcessor.from_pretrained(MODEL / "processor")
    tok = proc.tokenizer
    step = MiniMaxH3Ref2VATextEncoderStep()
    for it in todo:
        p = cache / f"{it['name']}.text.safetensors"
        if p.exists():
            continue
        t0 = time.time()
        refs = [MiniMaxH3VideoReference(frames=load_clip(it["proxy"], (PH, PW)), fps=24.0),
                MiniMaxH3ImageReference(image=Image.fromarray(load_image(it["first"], PIC_HW)))]
        vis, ic, vc, vt = step._gather_vision_features(proc, refs, 24.0)
        ids, tags = step._build_presentation(tok, prompt_for(it["caption"]), refs, ic, vc, vt)
        with torch.no_grad():
            hid = get_qwen3vl_prompt_embeds(te, proc, ids, vis, 50, dev, torch.bfloat16)[0].cpu()
        atomic_save(p, {"hidden": hid, "tags": torch.tensor(tags, dtype=torch.int64)}, {"caption": it["caption"]})
        print(it["name"], tuple(hid.shape), f"{time.time() - t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vae", "text"], required=True)
    ap.add_argument("--data", required=True, help="paired dataset directory (dataset.jsonl, refs/, first/, videos/)")
    ap.add_argument("--cache", required=True, help="output directory of the encodings")
    ap.add_argument("--ids", help="file with the idx values to encode (default: every row)")
    ap.add_argument("--no-target", action="store_true", help="skip the real video (test sets)")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_PROCID", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_NTASKS", 1)))
    ap.add_argument("--names", default="", help="comma list or @file: only these")
    ap.add_argument("--list", action="store_true", help="print names whose vae+text caches exist and exit")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    todo = items(a.data, a.ids, target=not a.no_target)
    if a.names:
        keep = set((open(a.names[1:]).read() if a.names.startswith("@") else a.names).replace(",", " ").split())
        todo = [it for it in todo if it["name"] in keep]
    if a.list:
        for it in todo:
            if (cache / f"{it['name']}.vae.safetensors").exists() and (cache / f"{it['name']}.text.safetensors").exists():
                print(it["name"])
        return
    if a.limit:
        todo = todo[:a.limit]
    todo = todo[a.shard::a.nshards]
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)) if "LOCAL_RANK" in os.environ else 0)
    print(f"stage={a.stage} shard {a.shard}/{a.nshards}: {len(todo)} items", flush=True)
    (stage_vae if a.stage == "vae" else stage_text)(todo, cache, dev)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
