"""LoRA training of the MiniMax-H3 Ref2VA model on pre-encoded proxy / video pairs (preencode.py).

    torchrun --nproc-per-node 8 -m simpwm.h3.train --data DS --cache CACHE --out DIR [--ids FILE]
                                                   [--steps 500 --stop-at 250 --lr 1.4e-4 --warmup 25 --rank 32]

One sample per GPU (per SP group of --sp GPUs) per micro-step (global batch = world / sp * accum). Rectified flow in
H3's convention: model time t = 1 - sigma (1 = clean), x_t = t x0 + (1 - t) noise, target velocity x0 - noise
(data-ward), sigma = shift_12(u).
References (proxy video, first frame) at t = 0.999 as at inference; target latent frame 0 pinned to the first-frame
latent (t = 0.999, no loss); silence audio generated jointly (audio sigma = shift_3(u), no audio loss).
--act-offload: the checkpointed block inputs (one [seq, 5376] bf16 tensor per block) live on the CPU between forward
and backward.
Resumes from the newest lora_stepNNNNN.safetensors (+ optimizer state) in --out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from simpwm.h3 import model as h3model
from simpwm.h3.common import COND_T, MODEL, build_layout, items, patchify, row_timesteps, shift_sigma


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="paired dataset directory (dataset.jsonl)")
    ap.add_argument("--cache", required=True, help="preencode.py output")
    ap.add_argument("--ids", help="file with the training idx values (default: every row)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--stop-at", type=int, default=0, help="stop for good at this absolute step (schedule still --steps)")
    ap.add_argument("--max-steps-this-job", type=int, default=0, help="stop (resumable) after this many steps")
    ap.add_argument("--time-budget-h", type=float, default=0, help="stop and save before this many hours")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift", type=float, default=12.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--act-offload", choices=["on", "off"], default="on")
    ap.add_argument("--sp", type=int, default=1, help="sequence-parallel GPUs per sample (Ulysses, seqpar.py)")
    return ap.parse_args()


def load_sample(cache, name, silence):
    v = load_file(str(cache / f"{name}.vae.safetensors"))
    t = load_file(str(cache / f"{name}.text.safetensors"))
    return dict(x0=v["target"], proxy=v["proxy"], first=v["first"], picture=v["first"], hidden=t["hidden"], tags=t["tags"], audio=silence)


def loss_on(model, s, dev, gen, shift):
    lay = build_layout(s["tags"])
    if SP is not None:
        SP.set_seq(lay["position_ids"].shape[0])
    rpf = lay["rows_per_frame"]
    x0 = patchify(s["x0"]).to(dev)                                     # (47*220, 96)
    pin = patchify(s["first"]).to(dev)                                 # (220, 96)
    cond = torch.cat([patchify(s["proxy"]), patchify(s["picture"])]).to(dev)
    u = torch.rand((), generator=gen).item()
    sig, sig_a = shift_sigma(u, shift), shift_sigma(u, 3.0)
    n = torch.randn(x0.shape, generator=gen).to(dev)
    xt = (1 - sig) * x0 + sig * n
    xt[:rpf] = COND_T * pin + (1 - COND_T) * torch.randn(pin.shape, generator=gen).to(dev)
    cn = COND_T * cond + (1 - COND_T) * torch.randn(cond.shape, generator=gen).to(dev)
    a0 = s["audio"].to(dev)
    at = (1 - sig_a) * a0 + sig_a * torch.randn(a0.shape, generator=gen).to(dev)
    uniq, inv = row_timesteps(lay, s["hidden"].shape[0], 1 - sig, 1 - sig_a, rpf)
    out = model(hidden_states=torch.cat([cn, xt])[None], audio_hidden_states=at[None],
                encoder_hidden_states=s["hidden"].to(dev)[None], timestep=uniq.to(dev), timestep_indices=inv.to(dev),
                token_tags=lay["token_tags"].to(dev), position_ids=lay["position_ids"].to(dev),
                video_indices=lay["video_indices"].to(dev), audio_indices=lay["audio_indices"].to(dev),
                text_indices=lay["text_indices"].to(dev), return_dict=False)[0]
    pred = out[0, lay["n_cond_video"] + rpf:].float()
    tgt = (x0 - n)[rpf:]
    return ((pred - tgt) ** 2).mean(), lay["position_ids"].shape[0], sig


SP = None


def install_ckpt(model, offload: bool):
    """Per-block activation checkpointing (non-reentrant); with offload the input of every DiT block is kept on the CPU
    (each block moves its input to the GPU and its output back, except the last one)."""
    from torch.utils.checkpoint import checkpoint
    blocks = list(model.transformer_blocks)
    ids = {id(b) for b in blocks}
    last = blocks[-1]

    def fn(block, *args):
        if not offload or id(block) not in ids:
            return checkpoint(block, *args, use_reentrant=False)
        dev = args[1].device                                            # temb lives on the GPU

        def run(h, *rest):
            out = block(h.to(dev, non_blocking=True), *rest)
            return out if block is last else out.to("cpu")
        return checkpoint(run, *args, use_reentrant=False)

    model._gradient_checkpointing_func = fn


def main():
    a = parse()
    rank, world, local, mesh = h3model.init_dist()
    dev = torch.device("cuda", local)
    log = (lambda *m: print(f"[{time.strftime('%H:%M:%S')}]", *m, flush=True)) if rank == 0 else (lambda *m: None)
    os.makedirs(a.out, exist_ok=True)
    cache = Path(a.cache)
    every = [it["name"] for it in items(a.data, a.ids)]
    names = [n for n in every if (cache / f"{n}.text.safetensors").exists() and (cache / f"{n}.vae.safetensors").exists()]
    if a.limit:
        names = names[:a.limit]
    else:  # never train silently on a partially encoded set
        assert len(names) == len(every), f"only {len(names)}/{len(every)} training samples encoded"
    log(f"world={world} accum={a.accum} sp={a.sp} global_batch={world // a.sp * a.accum} train={len(names)} steps={a.steps} lr={a.lr} rank={a.rank}")
    model, mods = h3model.build(str(MODEL), mesh, lora_rank=a.rank, grad_ckpt=True, log=log)
    install_ckpt(model, a.act_offload == "on")
    global SP
    sp = a.sp
    assert world % sp == 0
    slot, nslots = rank // sp, world // sp
    if sp > 1:
        from simpwm.h3 import seqpar as h3sp
        groups = [dist.new_group(list(range(g * sp, (g + 1) * sp))) for g in range(nslots)]
        SP = h3sp.install(model, groups[slot], sp, rank % sp)
    log(f"sp={sp} -> {nslots} samples per micro-step, global batch {nslots * a.accum}")
    params = [p for m in mods.values() for p in (m.lora_A, m.lora_B)]
    for p in params:
        p.requires_grad_(True)
    log(f"LoRA params {sum(p.numel() for p in params) / 1e6:.1f}M, mem {torch.cuda.memory_allocated() / 2**30:.1f}G")
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / a.warmup) * max(0.0, 1 - i / a.steps) if a.steps else 1.0)
    done = sorted(int(os.path.basename(p)[9:14]) for p in glob.glob(os.path.join(a.out, "lora_step*.safetensors")))
    start = done[-1] if done else 0
    if start:
        h3model.load_lora(mods, os.path.join(a.out, f"lora_step{start:05d}.safetensors"))
        op = os.path.join(a.out, f"opt_step{start:05d}.pt")
        if os.path.exists(op):
            opt.load_state_dict(torch.load(op, map_location=dev))
        for _ in range(start):
            sched.step()
        log(f"resumed at step {start}")
    silence = load_file(str(cache / "silence_audio.safetensors"))["audio"]
    gen = torch.Generator().manual_seed(a.seed * 100003 + slot * 7919 + start)     # identical within an SP group
    rng = random.Random(a.seed)
    order = []
    while len(order) < (a.steps + 1) * nslots * a.accum:
        e = names[:]
        rng.shuffle(e)
        order += e
    model.train()
    t0, tj = time.time(), time.time()
    for step in range(start, min(a.steps, a.stop_at) if a.stop_at else a.steps):
        lsum = 0.0
        for k in range(a.accum):
            idx = (step * a.accum + k) * nslots + slot
            s = load_sample(cache, order[idx], silence)
            loss, seq, sig = loss_on(model, s, dev, gen, a.shift)
            (loss / a.accum).backward()
            lsum += loss.item() / a.accum
        flat = torch.cat([p.grad.reshape(-1) for p in params])
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        if sp > 1:
            flat.mul_(sp)                       # per-sample gradient = sum over its SP group
        o = 0
        for p in params:
            n = p.numel()
            p.grad.copy_(flat[o:o + n].view_as(p))
            o += n
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        st = torch.tensor([lsum], device=dev)
        dist.all_reduce(st, op=dist.ReduceOp.AVG)
        n_done = step + 1 - start
        el = time.time() - t0
        log(f"step {step + 1}/{a.steps} loss={st.item():.4f} gnorm={float(gn):.3f} lr={sched.get_last_lr()[0]:.2e} "
            f"seq={seq} sigma={sig:.3f} mem={torch.cuda.max_memory_allocated() / 2**30:.1f}G "
            f"{el / n_done:.1f}s/step eta={(a.steps - step - 1) * el / n_done / 3600:.2f}h")
        if rank == 0:
            with open(os.path.join(a.out, "train_log.jsonl"), "a") as fh:
                fh.write(json.dumps(dict(step=step + 1, loss=st.item(), gnorm=float(gn), lr=sched.get_last_lr()[0], sec=el)) + "\n")
        stop = (a.stop_at and step + 1 >= a.stop_at) or (a.max_steps_this_job and n_done >= a.max_steps_this_job) or \
               (a.time_budget_h and (time.time() - tj + 1.5 * el / n_done) / 3600 > a.time_budget_h)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps or stop:
            if rank == 0:
                h3model.save_lora_ckpt(mods, os.path.join(a.out, f"lora_step{step + 1:05d}.safetensors"),
                                       dict(step=step + 1, rank=a.rank, lr=a.lr, world=world, accum=a.accum, sp=sp))
                torch.save(opt.state_dict(), os.path.join(a.out, f"opt_step{step + 1:05d}.pt"))
                log("saved", step + 1)
            dist.barrier()
        if stop:
            log("stopping for resubmission")
            break
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
