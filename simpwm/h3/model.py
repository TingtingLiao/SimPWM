"""The official MiniMax-H3 Ref2VA transformer (diffusers port), FSDP2-sharded over the GPUs of each node, with a
replicated LoRA (rank r on attn.to_q/to_k/to_v/to_out.0 and ff.net.0.proj/ff.net.2 of the 50 blocks).

The frozen blocks are sharded over the GPUs of a node (FSDP2, fully_shard with ignored_params); the LoRA stays outside
the sharding and train.py all-reduces its gradients by hand.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file

LORA_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")
FP32_MODULES = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank, self.scale = rank, alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        self.enabled = True

    def reset(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        dt = x.dtype
        return y + (x @ self.lora_A.to(dt).t()) @ (self.lora_B.to(dt) * self.scale).t()


def get_submodule_parent(root, name):
    parts = name.split(".")
    m = root
    for p in parts[:-1]:
        m = getattr(m, p) if not p.isdigit() else m[int(p)]
    return m, parts[-1]


def add_lora(model, rank: int, alpha: float):
    mods = {}
    for i, blk in enumerate(model.transformer_blocks):
        for t in LORA_TARGETS:
            parent, attr = get_submodule_parent(blk, t)
            base = parent[int(attr)] if attr.isdigit() else getattr(parent, attr)
            l = LoRALinear(base, rank, alpha)
            if attr.isdigit():
                parent[int(attr)] = l
            else:
                setattr(parent, attr, l)
            mods[f"transformer_blocks.{i}.{t}"] = l
    return mods


def lora_state(mods):
    sd = {}
    for n, m in mods.items():
        sd[f"{n}.lora_A"] = m.lora_A.detach().float().cpu()
        sd[f"{n}.lora_B"] = m.lora_B.detach().float().cpu()
    return sd


def load_lora(mods, path):
    sd = load_file(str(path))
    for n, m in mods.items():
        m.lora_A.data.copy_(sd[f"{n}.lora_A"])
        m.lora_B.data.copy_(sd[f"{n}.lora_B"])


def set_lora(mods, on: bool):
    for m in mods.values():
        m.enabled = on


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    lw = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("cuda", (world // lw, lw), mesh_dim_names=("rep", "shard"))
    return rank, world, local, mesh


def build(model_dir, mesh, lora_rank=0, lora_alpha=None, grad_ckpt=False, log=print):
    """Meta-init, LoRA, FSDP2 shard of every block (refiner + 50 DiT blocks), then load each rank's shard."""
    from diffusers import MiniMaxH3Transformer3DModel
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, distribute_tensor
    dev = torch.device("cuda", torch.cuda.current_device())
    cfg = json.load(open(os.path.join(model_dir, "transformer", "config.json")))
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**cfg)
    model.to(torch.bfloat16)
    for n in FP32_MODULES:
        getattr(model, n).float()
    model.requires_grad_(False)
    mods = {}
    if lora_rank:
        with torch.device("meta"):
            mods = add_lora(model, lora_rank, lora_alpha or lora_rank)
    lora_params = {p for m in mods.values() for p in (m.lora_A, m.lora_B)}
    shard_mesh = mesh["shard"] if mesh.ndim == 2 and mesh.shape[0] == 1 else mesh   # HSDP when >1 node
    blocks = list(model.token_refiner.refiner_blocks) + list(model.transformer_blocks)
    for b in blocks:
        fully_shard(b, mesh=shard_mesh, reshard_after_forward=True,
                    ignored_params={p for p in b.parameters() if p in lora_params})
    for i in range(len(blocks) - 1):
        blocks[i].set_modules_to_forward_prefetch([blocks[i + 1]])
    for i in range(1, len(blocks)):
        blocks[i].set_modules_to_backward_prefetch([blocks[i - 1]])
    root_direct = {p for p in model.parameters() if not isinstance(p, DTensor)}
    fully_shard(model, mesh=shard_mesh, ignored_params=root_direct)
    model.to_empty(device=dev)
    # rope buffer (computed, non-persistent)
    rf = model.rope.rope_freq_dim
    model.rope.inv_freq = 1.0 / (model.config.rope_theta ** (torch.arange(0, 2 * rf, 2, dtype=torch.float32, device=dev) / (2 * rf)))
    # weights
    t0 = time.time()
    files = sorted(glob.glob(os.path.join(model_dir, "transformer", "*.safetensors")))
    where = {}
    for f in files:
        with safe_open(f, "pt") as h:
            for k in h.keys():
                where[k] = f
    params = dict(model.named_parameters())
    missing = []
    handles = {}
    for name, p in params.items():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            continue
        key = name.replace(".base.", ".")
        if key not in where:
            missing.append(key)
            continue
        f = where[key]
        if f not in handles:
            handles[f] = safe_open(f, "pt", device="cpu")
        full = handles[f].get_tensor(key)
        if isinstance(p, DTensor):
            dt = distribute_tensor(full.to(dev, p.dtype), p.device_mesh, p.placements, src_data_rank=None)
            p.to_local().copy_(dt.to_local())
        else:
            p.data.copy_(full.to(dev, p.dtype))
    assert not missing, missing[:10]
    del handles
    torch.manual_seed(0)  # the same LoRA initialisation on every rank
    for m in mods.values():
        m.reset()
    log(f"loaded transformer in {time.time() - t0:.0f}s, mem={torch.cuda.memory_allocated() / 2**30:.1f}G")
    if grad_ckpt:
        model.enable_gradient_checkpointing()
    return model, mods


def save_lora_ckpt(mods, path, meta=None):
    tmp = f"{path}.tmp"
    save_file({k: v.contiguous() for k, v in lora_state(mods).items()}, tmp, metadata={k: str(v) for k, v in (meta or {}).items()})
    os.replace(tmp, path)
