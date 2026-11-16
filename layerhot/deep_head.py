"""Acceptance route, Stage 1a: exit heads that also read the previous token's deep state.

Guessing the token after y (position p) inside the layer-hot sweep, the loop has:
  - h_E(p): y's residual stream after E layers (this sweep)            -> "shallow"
  - f(p-1): the previous position's last-layer state (previous sweep)  -> "deep"
  - m(p-1): the previous position's layer-18 state (previous sweep)    -> "mid"
  - e(y):   y's input embedding                                         -> "emb" (plain EAGLE input)

Head: per-input learned RMSNorm, concat, Linear(k*H -> H), residual SiLU MLP (H -> 2H -> H), then the
model's frozen final norm and frozen reduced-vocab head (16k most frequent output tokens) + abstain.
Trained with cross-entropy on the full model's greedy token.

Inputs come from data/4B (layerhot.collect with --exits 1 2 18). Row i of a sequence uses
shallow/emb at i and deep/mid at i-1, so the first stored position of every sequence is skipped.

Usage: python -m layerhot.deep_head --inputs shallow1 deep mid [--epochs 6]
"""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from bench import snapshot
from bench.acceptance import token_classes
from layerhot.exit_head import Data, map_labels, to_mx
from layerhot.model_utils import MODELS, ReducedHead, load

CYCLE_SUSTAINED = 1.36  # one-guess cycle cost in one-row steps, chained loop under sustained load
CYCLE_GPU = 1.09  # same cycle, GPU work only (bench/cycle_parts.py)


class DeepHead(nn.Module):
    def __init__(self, s, rows: np.ndarray, n_inputs: int):
        super().__init__()
        H = s.hidden
        self.norms = [nn.RMSNorm(H, eps=s.model.args.rms_norm_eps) for _ in range(n_inputs)]
        self.fc = nn.Linear(n_inputs * H, H, bias=False)
        self.up = nn.Linear(H, 2 * H, bias=False)
        self.gate = nn.Linear(H, 2 * H, bias=False)
        self.down = nn.Linear(2 * H, H, bias=False)
        self.down.weight = mx.zeros_like(self.down.weight)  # MLP starts as identity residual
        self.w_abs = mx.zeros((H,))
        self.b_abs = mx.array([-2.0])
        self.vocab_rows = mx.array(rows.astype(np.int32))
        self.freeze(keys=["vocab_rows"])
        self._final_norm = s.inner.norm
        self._head = ReducedHead.from_model(s, self.vocab_rows)

    def __call__(self, xs: list[mx.array]) -> mx.array:
        x = self.fc(mx.concatenate([n(x.astype(mx.float32)) for n, x in zip(self.norms, xs)], axis=-1))
        x = x + self.down(nn.silu(self.gate(x)) * self.up(x))
        n = self._final_norm(x.astype(mx.bfloat16))
        lv = self._head(n).astype(mx.float32)
        la = (n.astype(mx.float32) @ self.w_abs)[..., None] + self.b_abs
        return mx.concatenate([lv, la], axis=-1)

    def guess(self, xs):
        p = mx.softmax(self(xs), axis=-1)
        idx = mx.argmax(p, axis=-1)
        V = self.vocab_rows.shape[0]
        return mx.where(idx == V, -1, self.vocab_rows[mx.minimum(idx, V - 1)]), mx.max(p, axis=-1)


def build_index(data: Data):
    """Positions usable as (current = i, previous = i-1) pairs within one sequence."""
    prompt = data.meta["prompt"]
    first = np.r_[True, prompt[1:] != prompt[:-1]]
    ok = ~first
    cur = np.nonzero(ok)[0]
    return cur, cur - 1


def features(s, data: Data, names: list[str]):
    cur, prev = build_index(data)
    src = {"shallow1": (1, cur), "shallow2": (2, cur), "deep": (data.info["n_layers"], prev), "mid": (18, prev)}
    feats = []
    for nm in names:
        if nm == "emb":
            feats.append(("emb", data.meta["input"][cur]))
        else:
            layer, idx = src[nm]
            feats.append((nm, data.h(layer), idx))
    return cur, feats


def gather(s, feats, sel):
    out = []
    for f in feats:
        if f[0] == "emb":
            out.append(s.embed(mx.array(f[1][sel])).astype(mx.bfloat16))
        else:
            out.append(to_mx(f[1][np.sort(f[2][sel])][np.argsort(np.argsort(f[2][sel]))]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["shallow1", "deep", "mid"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--train-frac", type=float, default=1.0, help="learning curve: use this share of training prompts")
    a = ap.parse_args(argv)
    s = load("4B")
    data = Data("4B")
    cur, feats = features(s, data, a.inputs)
    is_test = data.is_test[cur]
    tr, te = np.nonzero(~is_test)[0], np.nonzero(is_test)[0]
    if a.train_frac < 1.0:  # subsample whole training prompts, not positions
        pids = np.unique(data.meta["prompt"][cur][tr])
        keep = set(np.random.default_rng(1).permutation(pids)[: int(len(pids) * a.train_frac)].tolist())
        tr = tr[np.isin(data.meta["prompt"][cur][tr], list(keep))]
    rows = data.vocab(a.vocab)
    y_all = map_labels(data.label, rows)[cur]
    head = DeepHead(s, rows, len(a.inputs))
    params = head.trainable_parameters()
    steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
    opt = optim.AdamW(learning_rate=optim.cosine_decay(a.lr, steps), weight_decay=0.01)

    def loss_fn(p, xs, y):
        head.update(p)
        return nn.losses.cross_entropy(head(xs), y, reduction="mean")

    vg = mx.value_and_grad(loss_fn)

    def evaluate():
        toks, confs = [], []
        for i in range(0, len(te), 2048):
            sel = te[i:i + 2048]
            t, c = head.guess(gather(s, feats, sel))
            mx.eval(t, c)
            toks.append(np.array(t)); confs.append(np.array(c.astype(mx.float32)))
        return np.concatenate(toks), np.concatenate(confs)

    out = snapshot.new_results_dir("deep_head_" + "+".join(a.inputs) + (f"_frac{a.train_frac}" if a.train_frac < 1 else ""))
    snapshot.write_env(out, {"short": snapshot.short_snapshot(), "software": snapshot.software(),
                             "models": [snapshot.model_info(MODELS["4B"])], "config": vars(a)})
    rng = np.random.default_rng(0)
    t0 = time.time()
    hist = []
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), a.batch):
            sel = perm[i:i + a.batch]
            loss, g = vg(params, gather(s, feats, sel), mx.array(y_all[sel]))
            params = opt.apply_gradients(g, params)
            mx.eval(params, opt.state, loss)
            tot += loss.item() * len(sel)
        head.update(params)
        tok, conf = evaluate()
        acc = float((tok == data.label[cur][te]).mean())
        hist.append({"epoch": ep, "train_loss": tot / len(perm), "top1": acc})
        print(f"{'+'.join(a.inputs)} epoch {ep} loss {tot / len(perm):.3f} held-out top1 {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)

    tok, conf = evaluate()
    lab = data.label[cur][te]
    hit = tok == lab
    cls = token_classes(s.tokenizer, lab, data.meta["input"][cur][te])
    kind = data.kind[cur][te]
    acc = float(hit.mean())
    res = {"inputs": a.inputs, "train_positions": int(len(tr)), "test_positions": int(len(te)),
           "top1": acc, "history": hist,
           "by_class": {c: float(hit[cls == c].mean()) for c in np.unique(cls)},
           "by_kind": {k: float(hit[kind == k].mean()) for k in np.unique(kind)},
           "pr_curve": [{"tau": t, "coverage": float(((tok >= 0) & (conf >= t)).mean()),
                         "precision": float(hit[(tok >= 0) & (conf >= t)].mean()) if ((tok >= 0) & (conf >= t)).any() else None}
                        for t in (0.0, 0.3, 0.5, 0.7, 0.9)],
           "pred_speed_one_guess": {"tokens_per_cycle": 1 + acc,
                                    "at_sustained_cycle_cost": (1 + acc) / CYCLE_SUSTAINED,
                                    "at_gpu_only_cycle_cost": (1 + acc) / CYCLE_GPU}}
    (out / "deep_head.json").write_text(json.dumps(res, indent=2))
    mx.savez(str(out / "head.npz"), **dict(__import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(head.parameters())))
    print(json.dumps({k: v for k, v in res.items() if k not in ("history",)}, indent=2))
    print("results:", out)


if __name__ == "__main__":
    main()
