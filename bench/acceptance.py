"""Step 2: a(E), how often a shallow exit guesses the full model's own next token.

On the held-out prompts (split by prompt), for each exit E:
  - top-1 agreement of the logit lens and the trained affine exit head
  - agreement split by the class of the full model's token
  - precision vs coverage as the confidence threshold moves
  - a replay of the step 3 loop on the real sequences (chained guesses, stop at the
    first abstain or low-confidence guess, accept the matching prefix + 1 bonus token),
    giving tokens per sweep, and a predicted speedup from the step 1 timings

Usage: python -m bench.acceptance --model 0.6B --row-cost results/<ts>-row_cost_mlx
"""

from __future__ import annotations

import argparse
import json
import statistics
import string
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from bench import snapshot
from layerhot.exit_head import AffineExit, Data, evaluate_affine, to_mx, train_affine
from layerhot.model_utils import MODELS, load

FUNCTION_WORDS = set("""a an the of to in on at by for with from as and or but if then than that this these
those is are was were be been being it its it's he she they we you i me my your our their his her him them
not no so do does did have has had will would can could should may might must there here which who whom
what when where why how all any each some such into over under about after before between through during
up down out off again also just only very more most other same own too s t""".split())

THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
KMAX = [2, 3, 4, 8]


# ------------------------------------------------------------------ token classes

def token_classes(tok, ids_next: np.ndarray, ids_prev: np.ndarray) -> np.ndarray:
    cache: dict[int, str] = {}

    def dec(i: int) -> str:
        if i not in cache:
            cache[i] = tok.decode([int(i)])
        return cache[i]

    punct = set(string.punctuation) | {"’", "“", "”", "—", "–", "…"}
    out = []
    for n, p in zip(ids_next, ids_prev):
        s = dec(n)
        st = s.strip()
        if not st or all((c in punct) or not c.isalnum() for c in st):
            out.append("punct/ws")
        elif st.lower() in FUNCTION_WORDS and (s[0].isspace() or not dec(p)[-1:].isalnum()):
            out.append("function")
        elif s[0].isalnum() and dec(p)[-1:].isalnum():
            out.append("subword")
        else:
            out.append("content")
    return np.array(out)


# ------------------------------------------------------------------ cost model

class Cost:
    """Per-cycle time in units of one plain 1-row sweep, from step 1 measurements."""

    def __init__(self, raw_rows: list[dict], model: str, ctx: int, n_layers: int,
                 draft_head_ms: float):
        keep = [r for r in raw_rows if r["model"] == model and r["ctx"] == ctx and not r["discard"]]
        self.discarded = not keep
        if not keep:
            keep = [r for r in raw_rows if r["model"] == model and r["ctx"] == ctx]
        g: dict = {}
        for r in keep:
            g.setdefault((r["kind"], r["K"]), []).append(r["median_ms"])
        m = {k: statistics.median(v) for k, v in g.items()}
        self.m = m
        self.N = n_layers
        self.sweep1 = m[("sweep", 1)]
        self.draft_head_ms = draft_head_ms
        es = sorted(int(k[0][8:]) for k in m if k[0].startswith("layers_E"))
        ys = [m[(f"layers_E{e}", 1)] for e in es]
        self.e_fit = np.polyfit(es, ys, 1) if len(es) > 1 else (0.0, ys[0])

    def t_E(self, E: int) -> float:
        k = (f"layers_E{E}", 1)
        return self.m[k] if k in self.m else float(np.polyval(self.e_fit, E))

    def _interp(self, kind: str, K: int) -> float:
        ks = sorted(k for (kd, k) in self.m if kd == kind)
        return float(np.interp(K, ks, [self.m[(kind, k)] for k in ks]))

    @property
    def sync_ms(self) -> float:
        """Host round-trip cost of one separately evaluated call: a 1-layer call minus
        the per-layer share of a full sweep."""
        return max(0.0, self.t_E(1) - self.m[("body", 1)] / self.N)

    def cycle(self, E: int, rows: int, head_calls: int, ideal: bool = False) -> float:
        body = self._interp("body", rows) * (self.N - E) / self.N
        head = self._interp("head_full", rows)
        tE, th = self.t_E(E), self.draft_head_ms
        if ideal:  # no host round trip per draft call (e.g. a fused llama.cpp graph)
            tE = max(tE - self.sync_ms, self.m[("body", 1)] * E / self.N)
            th = max(th - self.sync_ms, 0.02)
        ms = rows * tE + head_calls * th + body + head
        return ms / self.sweep1


# ------------------------------------------------------------------ replay

def replay(tok_guess: np.ndarray, conf: np.ndarray, label: np.ndarray, prompt: np.ndarray,
           tau: float, kmax: int):
    """Replay the loop over each held-out sequence. Arrays are aligned to positions."""
    cycles = []  # (tokens accepted, rows, head calls)
    starts = np.nonzero(np.r_[True, prompt[1:] != prompt[:-1]])[0]
    ends = np.r_[starts[1:], len(prompt)]
    for s0, s1 in zip(starts, ends):
        i = s0
        while i < s1:
            g, acc, ok, calls = 0, 0, True, 0
            j = i
            while g < kmax - 1 and j < s1:
                calls += 1
                if tok_guess[j] < 0 or conf[j] < tau:
                    break
                g += 1
                if ok and tok_guess[j] == label[j]:
                    acc += 1
                else:
                    ok = False
                j += 1
            cycles.append((acc + 1, g + 1, calls))
            i += acc + 1
    c = np.array(cycles)
    return c


def time_head(fn, x, iters=50) -> float:
    for _ in range(5):
        mx.eval(fn(x))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn(x))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6B")
    ap.add_argument("--exits", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--row-cost", required=True, help="results dir of bench.row_cost")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--adapter", action="store_true", help="also train/evaluate the step 4a adapter")
    ap.add_argument("--adapter-epochs", type=int, default=6)
    a = ap.parse_args(argv)

    out = snapshot.new_results_dir(f"acceptance_{a.model}")
    snapshot.write_env(out, snapshot.full_snapshot(10, [MODELS[a.model]]))
    s = load(a.model)
    data = Data(a.model)
    N = data.info["n_layers"]
    te = data.test_idx
    label = data.label[te]
    prompt = data.meta["prompt"][te]
    cls = token_classes(s.tokenizer, label, data.meta["input"][te])
    raw = json.loads((Path(a.row_cost) / "raw.json").read_text())

    report = {"model": a.model, "n_layers": N, "test_positions": int(len(te)),
              "train_positions": int(len(data.train_idx)),
              "sanity": {
                  "label_eq_generated": float((data.label == data.meta["gen_next"]).mean()),
                  "lens_at_N_eq_label": float((data.meta[f"lens_top_{N}"] == data.label).mean()),
                  "near_tie_share_margin_lt_0.1": float((data.meta["margin"] < 0.1).mean()),
              },
              "class_share": {c: float((cls == c).mean()) for c in np.unique(cls)},
              "kind_share": {k: float((data.kind[te] == k).mean()) for k in np.unique(data.kind[te])},
              "exits": {}}

    x1 = mx.random.normal((1, 1, s.hidden)).astype(mx.bfloat16)
    lens_ms = time_head(lambda x: s.head(s.norm(x)), x1)

    best = None
    for e in a.exits:
        if e >= N:
            continue
        rec = {}
        # logit lens
        lt, lp = data.meta[f"lens_top_{e}"][te], data.meta[f"lens_p_{e}"][te].astype(np.float32)
        # trained affine head
        hp = data.dir / "heads" / f"affine_E{e}_V{a.vocab}.npz"
        if hp.exists() and not a.retrain:
            head = AffineExit.load(s, hp)
            hinfo = json.loads(hp.with_suffix(".json").read_text())
        else:
            hp.parent.mkdir(exist_ok=True)
            head, hinfo = train_affine(s, data, e, a.vocab, a.epochs)
            head.save(hp)
            hp.with_suffix(".json").write_text(json.dumps(hinfo, indent=2))
        head.half()  # evaluate and time what the decoder runs
        ev = evaluate_affine(head, data, e)
        at, ap_ = ev["tok"], ev["conf"]
        aff_ms = time_head(lambda x: head(x), x1)

        heads = [("lens", lt, lp, lens_ms), ("affine", at, ap_, aff_ms)]
        if a.adapter:
            from layerhot.adapter import AdapterExit, evaluate_adapter, train_adapter

            dp = data.dir / "heads" / f"adapter_E{e}_V{a.vocab}.npz"
            if dp.exists() and not a.retrain:
                ad = AdapterExit.load(s, dp)
                rec["adapter_training"] = json.loads(dp.with_suffix(".json").read_text())
            else:
                ad, rec["adapter_training"] = train_adapter(s, data, e, a.vocab, a.adapter_epochs)
                ad.save(dp)
                dp.with_suffix(".json").write_text(json.dumps(rec["adapter_training"], indent=2))
            ad.half()
            ev_ad = evaluate_adapter(s, ad, data, e)
            ad.prefill(mx.random.normal((1, a.ctx, s.hidden)).astype(mx.bfloat16))

            def ad_step(x):
                t = ad.guess(x[:, 0])
                ad._cache.trim(1)
                return t
            heads.append(("adapter", ev_ad["tok"], ev_ad["conf"], time_head(ad_step, x1)))
        for name, gt, gc, hms in heads:
            hit = gt == label
            r = {"top1": float(hit.mean()), "head_ms": hms,
                 "by_class": {c: float(hit[cls == c].mean()) for c in np.unique(cls)},
                 "by_kind": {k: float(hit[data.kind[te] == k].mean()) for k in np.unique(data.kind[te])},
                 "pr_curve": [], "replay": []}
            for tau in THRESHOLDS:
                cov = (gt >= 0) & (gc >= tau)
                r["pr_curve"].append({"tau": tau, "coverage": float(cov.mean()),
                                      "precision": float(hit[cov].mean()) if cov.any() else None})
            cost = Cost(raw, a.model, a.ctx, N, hms)
            for kmax in KMAX:
                for tau in THRESHOLDS[1:]:
                    c = replay(gt, gc, label, prompt, tau, kmax)
                    tps = c[:, 0].mean()
                    cyc = np.mean([cost.cycle(e, int(rw), int(hc)) for _, rw, hc in c])
                    cyc_i = np.mean([cost.cycle(e, int(rw), int(hc), True) for _, rw, hc in c])
                    pred = tps / cyc
                    row = {"kmax": kmax, "tau": tau, "tokens_per_sweep": float(tps),
                           "rows_per_sweep": float(c[:, 1].mean()), "cost_per_sweep": float(cyc),
                           "pred_speedup": float(pred),
                           "pred_speedup_no_host_overhead": float(tps / cyc_i)}
                    r["replay"].append(row)
                    if best is None or pred > best["pred_speedup"]:
                        best = {"E": e, "head": name, **row}
            r["cost_from_discarded_rounds"] = cost.discarded
            rec[name] = r
        rec["affine_training"] = hinfo
        report["exits"][e] = rec
        print(f"E={e}: " + "  ".join(f"{k} top1={rec[k]['top1']:.3f}" for k in ("lens", "affine", "adapter")
                                     if k in rec) + "  "
              f"best pred speedup so far={best['pred_speedup']:.3f} ({best})", flush=True)

    report["best"] = best
    report["gate"] = "step 3" if best and best["pred_speedup"] >= 1.2 else "step 4a/4b first"
    (out / "acceptance.json").write_text(json.dumps(report, indent=2))
    print("results:", out)
    print(json.dumps({"sanity": report["sanity"], "best": best, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
