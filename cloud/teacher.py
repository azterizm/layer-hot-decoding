"""Batched teacher forcing of Qwen3-4B (bf16, Hugging Face) with forward hooks.

The hooks on ``model.model.layers[L - 1]`` capture the residual stream after L layers, before the
final norm, for L in CAPTURE. This matches ``layerhot/collect.py`` (Hugging Face's last
``hidden_states`` entry is after the final norm, so it is not used).

Rows follow ``layerhot.deep_head.build_index``: for current position i the head gets ``h_1[i]`` or
``embed(token[i])``, plus ``h_18[i-1]`` and ``h_36[i-1]``, and the label is the full model's argmax at i
(the token it picks after token[i]). Rows run from i = P (first generated token) to T - 2, so the
stored position P - 1 of ``collect.py`` is skipped, as ``build_index`` skips the first stored row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
CAPTURE = (1, 18, 36)


@dataclass
class Batch:
    ids: torch.Tensor  # [B, T] long, right-padded with 0
    lens: torch.Tensor  # [B] long
    label: torch.Tensor  # [B, T] long, -1 where no label was computed
    h: dict  # L -> [B, T, H] bf16, residual stream after L layers (pre-norm)
    hn: torch.Tensor  # [B, T, H] bf16, after the final norm


@dataclass
class Rows:
    """One row per current position i: inputs at i (current token) and at i - 1 (previous position)."""
    cur: torch.Tensor  # [N, H] residual after the first captured layer, at i
    tok: torch.Tensor  # [N] token id at i (for the embedding arm)
    mid: torch.Tensor  # [N, H] second captured layer, at i - 1
    deep: torch.Tensor  # [N, H] last captured layer, at i - 1
    y: torch.Tensor  # [N] label (token id) at i
    seq: torch.Tensor  # [N] batch index
    pos: torch.Tensor  # [N] i


class Teacher:
    def __init__(self, model=None, capture=CAPTURE, device="cuda", dtype=torch.bfloat16):
        if model is None:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=dtype,
                                                         attn_implementation="sdpa")
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.inner = self.model.model
        self.device = device
        self.capture = tuple(capture)
        cfg = self.model.config
        self.H = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.vocab_size = cfg.vocab_size
        self._caps: dict = {}
        self._hooks = [self.inner.layers[L - 1].register_forward_hook(self._hook(L)) for L in self.capture]
        self.embed_w = self.inner.embed_tokens.weight
        self.norm_w = self.inner.norm.weight
        self.lm_w = self.model.lm_head.weight
        self._lm32 = None

    def _hook(self, L):
        def f(_mod, _inp, out):
            self._caps[L] = out[0] if isinstance(out, tuple) else out
        return f

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids.long(), self.embed_w)

    def logits32(self, hn: torch.Tensor) -> torch.Tensor:
        """fp32 logits from post-norm states (bf16 inputs are exact in fp32/TF32)."""
        if self._lm32 is None:
            self._lm32 = self.lm_w.float()
        return hn.float() @ self._lm32.T

    @torch.no_grad()
    def run(self, seqs: list, label_start: list[int], chunk: int = 2048) -> Batch:
        """Teacher-force right-padded sequences. ``label_start[b]``: first position that gets a label;
        labels run up to len - 2 (the last position's argmax would be a token past the sequence)."""
        B = len(seqs)
        T = max(len(s) for s in seqs)
        ids = torch.zeros(B, T, dtype=torch.long)
        mask = torch.zeros(B, T, dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, :len(s)] = torch.as_tensor(np.asarray(s, dtype=np.int64))
            mask[b, :len(s)] = 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        self._caps.clear()
        hn = self.inner(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        lens = mask.sum(1)
        pos = torch.arange(T, device=self.device)
        start = torch.as_tensor(label_start, device=self.device)
        want = (pos[None] >= start[:, None]) & (pos[None] <= lens[:, None] - 2)
        b_i, t_i = want.nonzero(as_tuple=True)
        label = torch.full((B, T), -1, dtype=torch.long, device=self.device)
        for c in range(0, len(b_i), chunk):
            bb, tt = b_i[c:c + chunk], t_i[c:c + chunk]
            label[bb, tt] = self.logits32(hn[bb, tt]).argmax(-1)
        h = {L: self._caps[L] for L in self.capture}
        self._caps.clear()
        return Batch(ids=ids, lens=lens, label=label, h=h, hn=hn)

    def rows(self, b: Batch, row_start: list[int]) -> Rows:
        """Rows i in [row_start[b], len - 2] that have a label. row_start must be >= 1."""
        B, T = b.ids.shape
        pos = torch.arange(T, device=b.ids.device)
        start = torch.as_tensor(row_start, device=b.ids.device)
        ok = (b.label >= 0) & (pos[None] >= start[:, None])
        s, i = ok.nonzero(as_tuple=True)
        L1, Lm, Ld = self.capture
        return Rows(cur=b.h[L1][s, i], tok=b.ids[s, i], mid=b.h[Lm][s, i - 1], deep=b.h[Ld][s, i - 1],
                    y=b.label[s, i], seq=s, pos=i)


def plan_batches(lengths: np.ndarray, order: np.ndarray, max_tokens: int, window: int = 2048,
                 seed: int = 0) -> list[np.ndarray]:
    """Group sequences (in ``order``) into teacher batches of at most ``max_tokens`` padded tokens.
    Within each window of ``window`` sequences, sort by length to cut padding, then shuffle the
    batches of that window so training does not see lengths in order."""
    rng = np.random.default_rng(seed)
    out = []
    for w in range(0, len(order), window):
        idx = order[w:w + window]
        idx = idx[np.argsort(-lengths[idx], kind="stable")]
        batches, cur, tmax = [], [], 0
        for j in idx:
            t = int(lengths[j])
            if cur and max(tmax, t) * (len(cur) + 1) > max_tokens:
                batches.append(np.array(cur))
                cur, tmax = [], 0
            cur.append(j)
            tmax = max(tmax, t)
        if cur:
            batches.append(np.array(cur))
        for k in rng.permutation(len(batches)):
            out.append(batches[k])
    return out
