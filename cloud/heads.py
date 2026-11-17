"""PyTorch heads for Phase 0.

All heads end in the model's frozen final norm and a frozen V-row slice of its output layer, plus a
learned abstain logit, as ``layerhot.deep_head.DeepHead`` does. Labels outside the slice train the
abstain class; an abstain or out-of-slice guess counts as a miss.

| arch     | shape                                                                                   |
|----------|-----------------------------------------------------------------------------------------|
| A-parity | ``DeepHead`` ported: per-input RMSNorm, concat, Linear, one residual SwiGLU block (2H)  |
| A-large  | same with 4 residual blocks of width 4H (about 335M parameters)                         |
| B        | one decoder block, D=2048, 16 heads: causal self-attention over the current-token stream, |
|          | cross-attention to [h_18, h_36] of strictly earlier positions, SwiGLU MLP, Linear to H   |

Arms: ``h1`` reads the current token's layer-1 state, ``e`` its input embedding, ``deepmid`` neither.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ARMS = {"h1": ("cur", "deep", "mid"), "e": ("emb", "deep", "mid"), "deepmid": ("deep", "mid"),
        "deep": ("deep",)}  # "deep" only for the pilot's anchor check


class OutSlice:
    """Frozen final norm and V rows of the output layer, shared by every head (not a module, so it
    is not saved with each head)."""

    def __init__(self, norm_w: torch.Tensor, eps: float, lm_w: torch.Tensor, rows, vocab_size: int):
        dev = lm_w.device
        self.norm_w = norm_w.detach()
        self.eps = eps
        self.rows = torch.as_tensor(np.asarray(rows), dtype=torch.long, device=dev)
        self.W = lm_w.detach()[self.rows].contiguous()
        self.V = len(self.rows)
        self.lut = torch.full((vocab_size,), self.V, dtype=torch.long, device=dev)
        self.lut[self.rows] = torch.arange(self.V, device=dev)

    def map(self, y: torch.Tensor) -> torch.Tensor:
        """Token ids -> class index (V = abstain / out of slice)."""
        return self.lut[y]

    def token(self, idx: torch.Tensor) -> torch.Tensor:
        """Class index -> token id, -1 for abstain."""
        return torch.where(idx == self.V, -1, self.rows[idx.clamp(max=self.V - 1)])


def frozen_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Qwen3RMSNorm on a bf16 input, as DeepHead calls the model's norm on x.astype(bf16)."""
    xf = x.to(w.dtype).float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(w.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d: int, inner: int, zero_out: bool = True):
        super().__init__()
        self.up = nn.Linear(d, inner, bias=False)
        self.gate = nn.Linear(d, inner, bias=False)
        self.down = nn.Linear(inner, d, bias=False)
        if zero_out:
            nn.init.zeros_(self.down.weight)  # the residual block starts as the identity

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Head(nn.Module):
    def __init__(self, out: OutSlice, H: int):
        super().__init__()
        self._out = [out]  # a list, so the shared frozen slice is not registered or saved
        self.w_abs = nn.Parameter(torch.zeros(H))
        self.b_abs = nn.Parameter(torch.tensor([-2.0]))

    @property
    def out(self) -> OutSlice:
        return self._out[0]

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        o = self.out
        n = frozen_norm(x, o.norm_w, o.eps)
        lv = (n @ o.W.T).float()
        with torch.autocast(n.device.type, enabled=False):
            la = n.float() @ self.w_abs.float() + self.b_abs.float()
        return torch.cat([lv, la[:, None]], dim=-1)

    @torch.no_grad()
    def guess(self, logits: torch.Tensor):
        """(token id or -1, confidence) from logits over V + 1 classes."""
        p = torch.softmax(logits.float(), dim=-1)
        conf, idx = p.max(dim=-1)
        return self.out.token(idx), conf


class MLPHead(Head):
    def __init__(self, out: OutSlice, H: int, eps: float, n_in: int, inner: int, blocks: int):
        super().__init__(out, H)
        self.norms = nn.ModuleList([RMSNorm(H, eps) for _ in range(n_in)])
        self.fc = nn.Linear(n_in * H, H, bias=False)
        self.blocks = nn.ModuleList([SwiGLU(H, inner) for _ in range(blocks)])

    def body(self, xs: list[torch.Tensor]) -> torch.Tensor:
        x = self.fc(torch.cat([n(x) for n, x in zip(self.norms, xs)], dim=-1))
        for blk in self.blocks:
            x = x + blk(x)
        return x

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        return self.logits(self.body(xs))


def rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary embedding on [B, heads, T, d] at integer positions ``pos`` [T]."""
    d = x.shape[-1]
    inv = 1.0 / base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d)
    ang = pos.float()[:, None] * inv[None]
    cos, sin = ang.cos(), ang.sin()
    xf = x.float()
    x1, x2 = xf[..., : d // 2], xf[..., d // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class XHead(Head):
    """One decoder block. Query i attends to current-stream positions <= i and to the memory
    [h_18, h_36] of positions j < i (plus a learned null slot, so position 0 has a key)."""

    def __init__(self, out: OutSlice, H: int, eps: float, D: int = 2048, n_heads: int = 16,
                 mlp: int = 5632):
        super().__init__(out, H)
        self.D, self.nh, self.hd = D, n_heads, D // n_heads
        self.in_norm = RMSNorm(H, eps)
        self.in_proj = nn.Linear(H, D, bias=False)
        self.mem_norms = nn.ModuleList([RMSNorm(H, eps), RMSNorm(H, eps)])
        self.mem_proj = nn.Linear(2 * H, D, bias=False)
        self.sa_norm = RMSNorm(D, eps)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.sa_o = nn.Linear(D, D, bias=False)
        self.ca_norm = RMSNorm(D, eps)
        self.ca_q = nn.Linear(D, D, bias=False)
        self.ca_kv = nn.Linear(D, 2 * D, bias=False)
        self.ca_o = nn.Linear(D, D, bias=False)
        self.null_kv = nn.Parameter(torch.randn(2, n_heads, 1, self.hd) * 0.02)
        self.mlp_norm = RMSNorm(D, eps)
        self.mlp = SwiGLU(D, mlp)
        self.out_norm = RMSNorm(D, eps)
        self.out_proj = nn.Linear(D, H, bias=False)
        nn.init.zeros_(self.sa_o.weight)
        nn.init.zeros_(self.ca_o.weight)

    def body(self, cur: torch.Tensor, mid: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        """cur, mid, deep: [B, T, H], all indexed by position (mid/deep at j are h_18[j], h_36[j])."""
        B, T, _ = cur.shape
        nh, hd, D = self.nh, self.hd, self.D
        pos = torch.arange(T, device=cur.device)
        x = self.in_proj(self.in_norm(cur))
        q, k, v = self.qkv(self.sa_norm(x)).view(B, T, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        a = F.scaled_dot_product_attention(q, k, v.to(q.dtype), is_causal=True)
        x = x + self.sa_o(a.transpose(1, 2).reshape(B, T, D))
        m = self.mem_proj(torch.cat([self.mem_norms[0](mid), self.mem_norms[1](deep)], dim=-1))
        mk, mv = self.ca_kv(m).view(B, T, 2, nh, hd).permute(2, 0, 3, 1, 4)
        q2 = self.ca_q(self.ca_norm(x)).view(B, T, nh, hd).transpose(1, 2)
        q2, mk = rope(q2, pos), rope(mk, pos)
        nk, nv = (t.to(q2.dtype).expand(B, nh, 1, hd) for t in self.null_kv)
        K = torch.cat([nk, mk.to(q2.dtype)], dim=2)
        V = torch.cat([nv, mv.to(q2.dtype)], dim=2)
        allow = torch.ones(T, T + 1, dtype=torch.bool, device=cur.device)
        allow[:, 1:] = torch.ones(T, T, dtype=torch.bool, device=cur.device).tril(diagonal=-1)
        a2 = F.scaled_dot_product_attention(q2, K, V, attn_mask=allow)
        x = x + self.ca_o(a2.transpose(1, 2).reshape(B, T, D))
        x = x + self.mlp(self.mlp_norm(x))
        return self.out_proj(self.out_norm(x))

    def forward(self, cur, mid, deep, sel=None) -> torch.Tensor:
        """Logits at positions ``sel`` = (batch idx, position idx), or at every position."""
        y = self.body(cur, mid, deep)
        y = y[sel] if sel is not None else y.reshape(-1, y.shape[-1])
        return self.logits(y)


ARCHS = {
    "A-parity": {"kind": "mlp", "blocks": 1, "inner_mult": 2},
    "A-large": {"kind": "mlp", "blocks": 4, "inner_mult": 4},
    "B": {"kind": "x", "D": 2048, "n_heads": 16, "mlp": 5632},
}


def make(arch: str, arm: str, out: OutSlice, H: int, eps: float, seed: int, arch_cfg: dict | None = None):
    """Heads with the same arch, seed and input count get identical initial weights."""
    a = dict(ARCHS[arch]) if arch_cfg is None else dict(arch_cfg)
    torch.manual_seed(1000 * seed + 17)
    if a["kind"] == "mlp":
        return MLPHead(out, H, eps, n_in=len(ARMS[arm]), inner=a["inner_mult"] * H, blocks=a["blocks"])
    if arm == "deepmid":
        raise ValueError("B always reads a current-token stream")
    return XHead(out, H, eps, D=a["D"], n_heads=a["n_heads"], mlp=a["mlp"])


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warm-up, then cosine decay to 0 at ``total`` (mlx cosine_decay when warmup = 0)."""
    if warmup and step < warmup:
        return peak * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * 0.5 * (1 + math.cos(math.pi * t))
