"""Step 2 data: the target's own greedy output, plus hidden states at shallow layers.

1. Greedy-generate continuations for the prompt mix (batched, mlx-lm).
2. Teacher-force each prompt+continuation through the model one sequence at a time,
   and for every generated position j record:
     - h_E[j], the residual stream after layer E, for E in EXITS and E = N (final)
     - label[j] = argmax of the full model's logits at j (the token the full model picks next)
     - margin[j] = top-1 minus top-2 logit (near-tie check)
     - logit-lens top-1 and its probability for each E (zero-training exit head)
   Position j is "the last accepted token"; h_E[j] is what a draft at exit E sees.

Hidden states are stored as raw bfloat16 bits (uint16 memmaps), one file per layer.

Usage: python -m layerhot.collect --model 0.6B [--max-tokens 256] [--batch 32]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import batch_generate

from layerhot import prompts as P
from layerhot.model_utils import load

EXITS = [1, 2, 3, 4, 6, 8]
DATA = Path(__file__).resolve().parents[1] / "data"


def data_dir(model: str) -> Path:
    return DATA / model


def generate(s, items, max_tokens: int, batch: int, out: Path) -> list[dict]:
    path = out / "gen.jsonl"
    if path.exists():
        return [json.loads(l) for l in path.read_text().splitlines()]
    encoded = [P.encode(s.tokenizer, it) for it in items]
    t0 = time.time()
    rows = []
    for i in range(0, len(items), batch):
        r = batch_generate(s.model, s.tokenizer, encoded[i:i + batch], max_tokens=max_tokens,
                           return_token_ids=True, completion_batch_size=batch)
        for it, p, g in zip(items[i:i + batch], encoded[i:i + batch], r.token_ids):
            rows.append({**it, "prompt_ids": p, "gen_ids": g})
        n = sum(len(x["gen_ids"]) for x in rows)
        print(f"gen {len(rows)}/{len(items)} prompts, {n} tokens, {n / (time.time() - t0):.0f} tok/s",
              flush=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))
    return rows


def lens_top1(s, h: mx.array) -> tuple[mx.array, mx.array]:
    logits = s.head(s.norm(h)).astype(mx.float32)
    top = mx.argmax(logits, axis=-1)
    p = mx.exp(mx.max(logits, axis=-1) - mx.logsumexp(logits, axis=-1))
    return top, p


def teacher_force(s, rows: list[dict], out: Path, exits_req: list[int] | None = None) -> None:
    N = s.n_layers
    exits = [e for e in (exits_req or EXITS) if e < N] + [N]
    spans = []
    total = 0
    for r in rows:
        P_ = len(r["prompt_ids"])
        T = P_ + len(r["gen_ids"])
        n = max(0, T - 1 - (P_ - 1))  # positions P-1 .. T-2
        spans.append((total, n))
        total += n
    H = s.hidden
    mm = {e: np.lib.format.open_memmap(out / f"h_{e}.npy", mode="w+", dtype=np.uint16, shape=(total, H))
          for e in exits}
    meta = {k: np.zeros(total, dtype=np.int32) for k in ("label", "gen_next", "input", "prompt", "pos")}
    margin = np.zeros(total, dtype=np.float32)
    lens = {e: (np.zeros(total, dtype=np.int32), np.zeros(total, dtype=np.float16)) for e in exits}

    t0 = time.time()
    for ri, (r, (off, n)) in enumerate(zip(rows, spans)):
        if n == 0:
            continue
        ids = r["prompt_ids"] + r["gen_ids"]
        P_ = len(r["prompt_ids"])
        toks = mx.array([ids])
        h = s.embed(toks)
        prev = 0
        sl = slice(P_ - 1, len(ids) - 1)
        caps = {}
        for e in exits:
            h = s.layers(h, None, prev, e)
            prev = e
            caps[e] = h[0, sl]
        logits = s.head(s.norm(caps[N])).astype(mx.float32)
        top2 = mx.topk(logits, 2, axis=-1)
        top2 = mx.sort(top2, axis=-1)
        label = mx.argmax(logits, axis=-1)
        lt = {e: lens_top1(s, caps[e]) for e in exits}
        mx.eval(caps, label, top2, lt)
        for e in exits:
            mm[e][off:off + n] = np.array(caps[e].view(mx.uint16))
            lens[e][0][off:off + n] = np.array(lt[e][0])
            lens[e][1][off:off + n] = np.array(lt[e][1])
        meta["label"][off:off + n] = np.array(label)
        meta["gen_next"][off:off + n] = np.array(ids[P_:])
        meta["input"][off:off + n] = np.array(ids[P_ - 1:-1])
        meta["prompt"][off:off + n] = r["id"]
        meta["pos"][off:off + n] = np.arange(n)
        margin[off:off + n] = np.array(top2[:, 1] - top2[:, 0])
        if ri % 50 == 0:
            print(f"tf {ri}/{len(rows)} seqs, {off + n} positions, {time.time() - t0:.0f}s", flush=True)
    for m in mm.values():
        m.flush()
    np.savez(out / "meta.npz", margin=margin, **meta,
             **{f"lens_top_{e}": lens[e][0] for e in exits},
             **{f"lens_p_{e}": lens[e][1] for e in exits})
    split = {r["id"]: r["split"] for r in rows}
    kind = {r["id"]: r["kind"] for r in rows}
    (out / "prompts.json").write_text(json.dumps({"split": split, "kind": kind, "exits": exits,
                                                  "n_layers": N, "hidden": H, "total": total}))
    agree = float((meta["label"] == meta["gen_next"]).mean())
    print(f"positions={total}  label==generated: {agree:.4f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6B")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="only the first N prompts (smoke test)")
    ap.add_argument("--exits", type=int, nargs="+", default=EXITS)
    a = ap.parse_args(argv)
    out = data_dir(a.model if not a.limit else f"{a.model}-smoke")
    out.mkdir(parents=True, exist_ok=True)
    s = load(a.model)
    items = P.build()
    if a.limit:
        items = items[:a.limit]
    rows = generate(s, items, a.max_tokens, a.batch, out)
    teacher_force(s, rows, out, a.exits)


if __name__ == "__main__":
    main()
