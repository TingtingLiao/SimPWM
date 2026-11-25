"""Ulysses-style sequence parallelism for the MiniMax-H3 DiT blocks (diffusers port), sp ranks per sample.

Every rank of an SP group builds the full packed sequence (embeddings, text refiner, temb: cheap) and the DiT stack runs
on a contiguous slice of rows per rank (uneven split allowed). Inside attention, q/k/v of the local rows are exchanged
all-to-all so that each rank attends over ALL rows with 56/sp heads, then exchanged back. After the last block the rows
are all-gathered, so the output heads and the loss see the full sequence on every rank of the group.

Gradients: every rank of a group computes the same loss; the gather's backward returns the rank's own slice of the
(identical) output gradient, so each LoRA gradient is a partial sum over the rank's rows -> the full per-sample LoRA
gradient is the SUM over the SP group (train.py all-reduces with AVG over the world and multiplies by sp).
Hooks: a forward pre-hook on every DiT block slices the row-indexed args (hidden at block 0, adaln indices, rope),
a forward hook on the last block gathers; they also fire during checkpoint recomputation (symmetric on all ranks).
"""
from __future__ import annotations

import torch
import torch.distributed as dist


class _A2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, in_splits, out_splits, group):
        ctx.in_splits, ctx.out_splits, ctx.group = in_splits, out_splits, group
        out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), out_splits, in_splits, group=group)
        return out

    @staticmethod
    def backward(ctx, g):
        out = g.new_empty((sum(ctx.in_splits),) + tuple(g.shape[1:]))
        dist.all_to_all_single(out, g.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group)
        return out, None, None, None


class _Gather(torch.autograd.Function):
    """(B, n_r, C) local rows -> (B, S, C) on every rank; backward = own slice of the gradient."""

    @staticmethod
    def forward(ctx, x, sizes, rank, group):
        ctx.off, ctx.n = sum(sizes[:rank]), sizes[rank]
        m = max(sizes)
        buf = x.new_zeros((x.shape[0], m) + tuple(x.shape[2:]))
        buf[:, :x.shape[1]] = x
        parts = [torch.empty_like(buf) for _ in sizes]
        dist.all_gather(parts, buf.contiguous(), group=group)
        return torch.cat([p[:, :s] for p, s in zip(parts, sizes)], 1)

    @staticmethod
    def backward(ctx, g):
        return g[:, ctx.off:ctx.off + ctx.n].contiguous(), None, None, None


class SPState:
    def __init__(self, group, sp, rank_in_group):
        self.group, self.sp, self.r = group, sp, rank_in_group
        self.S = None

    def set_seq(self, S):
        self.S = S
        base, rem = divmod(S, self.sp)
        self.sizes = [base + (i < rem) for i in range(self.sp)]
        self.off = sum(self.sizes[:self.r])
        self.n = self.sizes[self.r]

    def loc(self, t, dim):
        return t.narrow(dim, self.off, self.n)


class SPAttnProcessor:
    """MiniMaxH3AttnProcessor with the all-to-all over heads around the attention kernel."""

    def __init__(self, st: SPState):
        self.st = st

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        st = self.st
        q = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        k = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        v = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if rotary_emb is not None:
            q = _apply_rotary_emb(q, *rotary_emb)
            k = _apply_rotary_emb(k, *rotary_emb)
        B, n, H, D = q.shape
        assert B == 1 and n == st.n and H % st.sp == 0
        hs = H // st.sp

        def fwd(x):          # (1, n_r, H, D) -> (1, S, H/sp, D)
            x = x[0].reshape(n, st.sp, hs, D).transpose(0, 1).reshape(st.sp * n, hs, D)
            return _A2A.apply(x, [n] * st.sp, st.sizes, st.group)[None]

        o = dispatch_attention_fn(fwd(q), fwd(k), fwd(v), attn_mask=None, dropout_p=0.0, is_causal=False)
        o = _A2A.apply(o[0].contiguous(), st.sizes, [n] * st.sp, st.group)          # (sp*n, hs, D)
        o = o.reshape(st.sp, n, hs, D).transpose(0, 1).reshape(1, n, H * D).type_as(q)
        o = attn.to_out[0](o)
        return attn.to_out[1](o)


def install(model, group, sp, rank_in_group) -> SPState:
    st = SPState(group, sp, rank_in_group)
    blocks = list(model.transformer_blocks)
    last = blocks[-1]

    def pre(mod, args):
        h, temb, adaln, rot = args[:4]
        if h.shape[1] == st.S:
            h = st.loc(h, 1)
        if adaln.shape[0] == st.S:
            adaln = st.loc(adaln, 0)
        if rot[0].shape[0] == st.S:
            rot = (st.loc(rot[0], 0), st.loc(rot[1], 0))
        return (h, temb, adaln, rot) + tuple(args[4:])

    def post(mod, args, out):
        return _Gather.apply(out, st.sizes, st.r, st.group)

    for b in blocks:
        b.register_forward_pre_hook(pre)
        b.attn.set_processor(SPAttnProcessor(st))
    last.register_forward_hook(post)
    return st
