"""Step 4b (offline): spend a spare row on the exit head's top-2 alternative.

Depth-1 tree per cycle, from the last accepted token's layer-E state:
  - p1 >= tau1          -> one guess (top-1): 2 rows
  - p1 + p2 >= tau2     -> two sibling guesses (top-1, top-2): 3 rows, siblings do not
                           attend to each other; accept whichever equals the full model's token
  - otherwise           -> no guess: 1 row
Tokens per sweep = 1 + hit. Cost from step 1 timings (same model as bench.acceptance).

Usage: python -m bench.tree_top2 --model 1.7B --exit 2 --row-cost results/<ts>-row_cost_mlx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from bench import snapshot
from bench.acceptance import Cost, time_head
from layerhot.exit_head import AffineExit, Data, to_mx
from layerhot.model_utils import load


def top2(head: AffineExit, data: Data, e: int, chunk: int = 2048):
    H = data.h(e)
    te = data.test_idx
    V = head.vocab_rows.shape[0]
    t1s, t2s, p1s, p2s = [], [], [], []
    for i in range(0, len(te), chunk):
        p = mx.softmax(head(to_mx(H[te[i:i + chunk]])), axis=-1)
        idx = mx.argsort(-p, axis=-1)[:, :2]
        pp = mx.take_along_axis(p, idx, axis=-1)
        tok = mx.where(idx == V, -1, head.vocab_rows[mx.minimum(idx, V - 1)])
        mx.eval(tok, pp)
        tok, pp = np.array(tok), np.array(pp.astype(mx.float32))
        t1s.append(tok[:, 0]); t2s.append(tok[:, 1]); p1s.append(pp[:, 0]); p2s.append(pp[:, 1])
    return [np.concatenate(x) for x in (t1s, t2s, p1s, p2s)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.7B")
    ap.add_argument("--exit", type=int, default=2)
    ap.add_argument("--row-cost", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    a = ap.parse_args(argv)
    s = load(a.model)
    data = Data(a.model)
    N = data.info["n_layers"]
    head = AffineExit.load(s, data.dir / "heads" / f"affine_E{a.exit}_V16384.npz").half()
    t1, t2, p1, p2 = top2(head, data, a.exit)
    y = data.label[data.test_idx]
    hms = time_head(lambda x: head(x), mx.random.normal((1, 1, s.hidden)).astype(mx.bfloat16))
    raw = json.loads((Path(a.row_cost) / "raw.json").read_text())
    cost = Cost(raw, a.model, a.ctx, N, hms)
    hit1, hit2 = (t1 == y), (t1 == y) | (t2 == y)
    res = {"model": a.model, "exit": a.exit, "top1": float(hit1.mean()), "top2": float(hit2.mean()),
           "grid": []}
    best = None
    for tau1 in (0.3, 0.5, 0.7, 0.9, 1.01):
        for tau2 in (0.3, 0.5, 0.7, 0.9, 1.01):
            one = (t1 >= 0) & (p1 >= tau1)
            two = ~one & (t1 >= 0) & (t2 >= 0) & (p1 + p2 >= tau2)
            tok = 1 + (one & hit1) + (two & hit2)
            rows = 1 + one + 2 * two
            # E layers: 1 call for row 0, 1 call (1 or 2 rows) for the guesses; one head call
            cyc = np.array([cost.cycle(a.exit, int(r), 1) for r in rows])
            # cost.cycle charges one E-layer call per row; a 2-sibling call is one call
            cyc = cyc - two * cost.t_E(a.exit) / cost.sweep1
            S = float(tok.mean() / cyc.mean())
            row = {"tau1": tau1, "tau2": tau2, "tokens_per_sweep": float(tok.mean()),
                   "rows_per_sweep": float(rows.mean()), "share_tree": float(two.mean()),
                   "pred_speedup": S}
            res["grid"].append(row)
            if best is None or S > best["pred_speedup"]:
                best = row
    res["best"] = best
    d = snapshot.new_results_dir(f"tree_top2_{a.model}_E{a.exit}")
    snapshot.write_env(d, snapshot.full_snapshot(10))
    out = d / "tree_top2.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("model", "exit", "top1", "top2", "best")}, indent=2))
    print("written", out)


if __name__ == "__main__":
    main()
