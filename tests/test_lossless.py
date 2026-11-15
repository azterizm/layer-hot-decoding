"""Step 3 check: the layer-hot loop emits exactly mlx-lm's greedy tokens.

Runs on held-out prompts. A mismatch is allowed only at a near-tie: the full model's
top-2 logits at the first diverging position (computed by plain single-row decode)
differ by at most NEAR_TIE_ULPS bf16 steps at the top logit's magnitude. Batched (multi-row)
verify writes KV entries with slightly different rounding than one-row decode, and these
differences accumulate; measured: one step of batching moves logits by up to 1 bf16 step
(0.25 at |logit| ~ 38 on 1.7B; 0.31 on 0.6B). Every mismatch is printed with its margin.

Env: LH_MODEL (default 0.6B), LH_EXIT (2), LH_KMAX (4), LH_TAU (0.5), LH_HEAD (affine|adapter|lens),
     LH_MODE (step|chain),
     LH_N (number of prompts, 24), LH_TOKENS (128)
"""

from __future__ import annotations

import os

import mlx.core as mx
import pytest

from layerhot import prompts as P
from layerhot.collect import data_dir
from layerhot.exit_head import AffineExit, LensExit
from layerhot.generate import LayerHotDecoder, mlx_greedy
from layerhot.model_utils import load

MODEL = os.environ.get("LH_MODEL", "0.6B")
EXIT = int(os.environ.get("LH_EXIT", "2"))
KMAX = int(os.environ.get("LH_KMAX", "4"))
TAU = float(os.environ.get("LH_TAU", "0.5"))
HEAD = os.environ.get("LH_HEAD", "affine")
MODE = os.environ.get("LH_MODE", "step")
N = int(os.environ.get("LH_N", "24"))
TOKENS = int(os.environ.get("LH_TOKENS", "128"))
NEAR_TIE_ULPS = 2


def bf16_ulp(x: float) -> float:
    import math

    return 2.0 ** (math.floor(math.log2(max(abs(x), 1e-30))) - 7)


@pytest.fixture(scope="module")
def setup():
    s = load(MODEL)
    heads = data_dir(MODEL) / "heads"
    if HEAD == "affine":
        head = AffineExit.load(s, heads / f"affine_E{EXIT}_V16384.npz").half()
    elif HEAD == "adapter":
        from layerhot.adapter import AdapterExit

        head = AdapterExit.load(s, heads / f"adapter_E{EXIT}_V16384.npz").half()
    else:
        head = LensExit(s)
    dec = LayerHotDecoder(s, head, EXIT, KMAX, TAU, mode=MODE)
    items = [it for it in P.build() if it["split"] == "test"][:N]
    return s, dec, items


def margin_at(s, prompt: list[int], prefix: list[int]) -> tuple[float, float]:
    """(top-1 minus top-2 logit, top-1 logit) of plain decoding after prompt + prefix, along
    mlx-lm's own path: prefill prompt[:-1], then one token at a time. (A single prefill of
    prompt + prefix is not the same computation; it moved logits by up to 0.45 on 4B.)"""
    cache = s.make_cache()
    s.forward(mx.array([prompt[:-1]]), cache)
    for t in [prompt[-1]] + prefix[:-1]:
        mx.eval(s.forward(mx.array([[t]]), cache))
    last = prefix[-1] if prefix else prompt[-1]
    if not prefix:  # undo the extra step: the loop above already consumed prompt[-1]
        s.trim(cache, 1)
    logits = s.forward(mx.array([[last]]), cache)[0, -1].astype(mx.float32)
    top = mx.sort(mx.topk(logits, 2))
    return float((top[1] - top[0]).item()), float(top[1].item())


def test_token_identical(setup):
    s, dec, items = setup
    bad, near, total_cycles, total_tokens = [], [], 0, 0
    for it in items:
        ids = P.encode(s.tokenizer, it)
        ref, _ = mlx_greedy(s, ids, TOKENS)
        out, st = dec.generate(ids, TOKENS)
        total_cycles += st.cycles
        total_tokens += st.tokens
        if out == ref:
            continue
        k = next((i for i, (a, b) in enumerate(zip(out, ref)) if a != b), min(len(out), len(ref)))
        m, top = margin_at(s, ids, ref[:k]) if k < len(ref) else (float("nan"), 0.0)
        ulps = m / bf16_ulp(top) if top else float("nan")
        rec = (it["id"], it["kind"], k, len(out), len(ref), m, round(ulps, 2))
        print(f"mismatch prompt={it['id']} ({it['kind']}) at token {k}: margin {m:.4f} "
              f"= {ulps:.1f} bf16 steps at top logit {top:.2f}")
        (near if ulps <= NEAR_TIE_ULPS else bad).append(rec)
    print(f"{len(items)} prompts, {total_tokens} tokens, {total_tokens / max(1, total_cycles):.3f} "
          f"tokens/sweep, near-tie mismatches={len(near)}, real mismatches={len(bad)}")
    assert not bad, f"non-tie mismatches: {bad}"
