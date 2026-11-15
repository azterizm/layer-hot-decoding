"""Per-layer access to mlx-lm Qwen3 models.

Splits the forward pass into embed -> layers[a:b] -> final norm -> LM head so the
bench, the collector and the decode loop all run the exact same ops as mlx-lm.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load as _load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

MODELS = {
    "0.6B": "mlx-community/Qwen3-0.6B-4bit",
    "1.7B": "mlx-community/Qwen3-1.7B-4bit",
    "4B": "mlx-community/Qwen3-4B-4bit",
}


@dataclass
class Split:
    model: nn.Module
    tokenizer: object

    @property
    def inner(self):
        return self.model.model

    @property
    def n_layers(self) -> int:
        return len(self.inner.layers)

    @property
    def hidden(self) -> int:
        return self.model.args.hidden_size

    def embed(self, tokens: mx.array) -> mx.array:
        return self.inner.embed_tokens(tokens)

    def layers(self, h: mx.array, cache, start: int, end: int) -> mx.array:
        """Run layers[start:end]. ``cache`` is the full per-layer list (or None)."""
        if cache is None:
            mask = create_attention_mask(h, None)
            for layer in self.inner.layers[start:end]:
                h = layer(h, mask, None)
            return h
        mask = create_attention_mask(h, cache[start])
        for i in range(start, end):
            h = self.inner.layers[i](h, mask, cache[i])
        return h

    def norm(self, h: mx.array) -> mx.array:
        return self.inner.norm(h)

    def head(self, h_normed: mx.array) -> mx.array:
        if self.model.args.tie_word_embeddings:
            return self.inner.embed_tokens.as_linear(h_normed)
        return self.model.lm_head(h_normed)

    def forward(self, tokens: mx.array, cache=None) -> mx.array:
        h = self.layers(self.embed(tokens), cache, 0, self.n_layers)
        return self.head(self.norm(h))

    def make_cache(self):
        return make_prompt_cache(self.model)

    @staticmethod
    def trim(cache, n: int, start: int = 0, end: int | None = None) -> None:
        if n <= 0:
            return
        if start == 0 and end is None:
            trim_prompt_cache(cache, n)
        else:
            for c in cache[start:end]:
                c.trim(n)

    def head_rows(self, rows: mx.array) -> "ReducedHead":
        """A reduced-vocabulary LM head made of selected output rows (no training)."""
        return ReducedHead.from_model(self, rows)


class ReducedHead(nn.Module):
    """LM head restricted to a subset of vocabulary rows, kept quantized if the source is."""

    def __init__(self, rows: mx.array, weight, scales=None, biases=None, group_size=64, bits=4,
                 mode="affine"):
        super().__init__()
        self.rows = rows
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

    @classmethod
    def from_model(cls, s: Split, rows: mx.array) -> "ReducedHead":
        src = s.inner.embed_tokens if s.model.args.tie_word_embeddings else s.model.lm_head
        if isinstance(src, (nn.QuantizedEmbedding, nn.QuantizedLinear)):
            return cls(rows, src.weight[rows], src.scales[rows],
                       None if getattr(src, "biases", None) is None else src.biases[rows],
                       src.group_size, src.bits, getattr(src, "mode", "affine"))
        return cls(rows, src.weight[rows])

    def __call__(self, x: mx.array) -> mx.array:
        if self.scales is None:
            return x @ self.weight.T
        return mx.quantized_matmul(x, self.weight, self.scales, self.biases, transpose=True,
                                   group_size=self.group_size, bits=self.bits, mode=self.mode)


def load(name: str) -> Split:
    repo = MODELS.get(name, name)
    model, tok = _load(repo)
    return Split(model, tok)
