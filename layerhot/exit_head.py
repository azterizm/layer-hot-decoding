"""Exit heads that guess the next token from the residual stream after layer E.

- ``LogitLens``: the model's own final norm + LM head on h_E. No training.
- ``AffineExit``: h -> A h + b (A starts at identity), then the frozen final norm and
  a frozen reduced-vocabulary head (the V most frequent tokens of the target's own
  output, rows taken from the model's quantized LM head), plus a learned "abstain"
  logit. Labels outside the reduced vocabulary train the abstain class.

Train: python -m layerhot.exit_head --model 0.6B --exit 2 [--vocab 16384]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from layerhot.collect import data_dir
from layerhot.model_utils import ReducedHead, load


# ------------------------------------------------------------------ data

class Data:
    def __init__(self, model: str):
        d = data_dir(model)
        self.dir = d
        info = json.loads((d / "prompts.json").read_text())
        self.info = info
        self.meta = dict(np.load(d / "meta.npz"))
        prompt = self.meta["prompt"]
        # the split comes from the current prompt builder (by topic group), checked by text
        from layerhot import prompts as P

        built = {it["id"]: it for it in P.build()}
        gen = [json.loads(l) for l in (d / "gen.jsonl").read_text().splitlines()]
        assert all(built[g["id"]]["text"] == g["text"] for g in gen), "prompt set changed"
        split = {g["id"]: built[g["id"]]["split"] for g in gen}
        kind = {int(k): v for k, v in info["kind"].items()}
        self.is_test = np.array([split[int(p)] == "test" for p in prompt])
        self.kind = np.array([kind[int(p)] for p in prompt])
        self.label = self.meta["label"]
        self.train_idx = np.nonzero(~self.is_test)[0]
        self.test_idx = np.nonzero(self.is_test)[0]

    def h(self, e: int) -> np.ndarray:
        return np.load(self.dir / f"h_{e}.npy", mmap_mode="r")

    def vocab(self, n: int) -> np.ndarray:
        """The n most frequent next tokens in the training split, most frequent first."""
        counts = np.bincount(self.label[self.train_idx])
        order = np.argsort(-counts, kind="stable")
        order = order[counts[order] > 0]
        return order[:n]


def to_mx(h_bits: np.ndarray) -> mx.array:
    return mx.array(np.ascontiguousarray(h_bits)).view(mx.bfloat16)


# ------------------------------------------------------------------ heads

class AffineExit(nn.Module):
    def __init__(self, s, rows: np.ndarray):
        super().__init__()
        H = s.hidden
        self.A = mx.eye(H, dtype=mx.float32)
        self.b = mx.zeros((H,), dtype=mx.float32)
        self.w_abs = mx.zeros((H,), dtype=mx.float32)
        self.b_abs = mx.array([-2.0], dtype=mx.float32)
        self.vocab_rows = mx.array(rows.astype(np.int32))
        # frozen parts
        self._norm = s.inner.norm
        self._head = ReducedHead.from_model(s, self.vocab_rows)
        self.freeze(keys=["vocab_rows"])

    def trainable(self):
        return {"A": self.A, "b": self.b, "w_abs": self.w_abs, "b_abs": self.b_abs}

    def __call__(self, h: mx.array) -> mx.array:
        """h: (..., H) bf16/float. Returns logits (..., V+1); the last column is abstain."""
        x = h.astype(self.A.dtype) @ self.A.T + self.b
        n = self._norm(x.astype(mx.bfloat16))
        lv = self._head(n).astype(mx.float32)
        la = (n.astype(mx.float32) @ self.w_abs)[..., None] + self.b_abs
        return mx.concatenate([lv, la], axis=-1)

    def guess(self, h: mx.array) -> tuple[mx.array, mx.array]:
        """Returns (token id or -1 for abstain, confidence)."""
        logits = self(h)
        p = mx.softmax(logits, axis=-1)
        idx = mx.argmax(p, axis=-1)
        conf = mx.max(p, axis=-1)
        V = self.vocab_rows.shape[0]
        tok = mx.where(idx == V, -1, self.vocab_rows[mx.minimum(idx, V - 1)])
        return tok, conf

    def half(self) -> "AffineExit":
        """bf16 affine map for decoding (halves the bytes read per guess)."""
        self.A = self.A.astype(mx.bfloat16)
        self.b = self.b.astype(mx.bfloat16)
        return self

    def save(self, path: Path) -> None:
        mx.savez(str(path), A=self.A, b=self.b, w_abs=self.w_abs, b_abs=self.b_abs,
                 vocab_rows=self.vocab_rows)

    @classmethod
    def load(cls, s, path: Path) -> "AffineExit":
        w = mx.load(str(path))
        m = cls(s, np.array(w["vocab_rows"]))
        m.A, m.b, m.w_abs, m.b_abs = w["A"], w["b"], w["w_abs"], w["b_abs"]
        return m


class LensExit(nn.Module):
    """Logit lens: the model's own final norm + LM head (optionally restricted to ``rows``)."""

    def __init__(self, s, rows: np.ndarray | None = None):
        super().__init__()
        self._s = s
        self.vocab_rows = None if rows is None else mx.array(rows.astype(np.int32))
        self._head = None if rows is None else ReducedHead.from_model(s, self.vocab_rows)

    def guess(self, h: mx.array) -> tuple[mx.array, mx.array]:
        n = self._s.norm(h)
        logits = (self._s.head(n) if self._head is None else self._head(n)).astype(mx.float32)
        idx = mx.argmax(logits, axis=-1)
        conf = mx.exp(mx.max(logits, axis=-1) - mx.logsumexp(logits, axis=-1))
        tok = idx if self.vocab_rows is None else self.vocab_rows[idx]
        return tok, conf


def map_labels(label: np.ndarray, rows: np.ndarray) -> np.ndarray:
    lut = np.full(int(max(label.max(), rows.max())) + 1, len(rows), dtype=np.int32)
    lut[rows] = np.arange(len(rows), dtype=np.int32)
    return lut[label]


def train_affine(s, data: Data, e: int, vocab: int = 16384, epochs: int = 4, batch: int = 256,
                 lr: float = 3e-4, log=print) -> tuple[AffineExit, dict]:
    rows = data.vocab(vocab)
    head = AffineExit(s, rows)
    H = data.h(e)
    y_all = map_labels(data.label, rows)
    tr = data.train_idx
    Xtr = to_mx(H[np.sort(tr)])
    ytr = mx.array(y_all[np.sort(tr)])
    mx.eval(Xtr, ytr)

    params = head.trainable()
    steps = epochs * ((len(tr) + batch - 1) // batch)
    sched = optim.cosine_decay(lr, steps)
    opt = optim.Adam(learning_rate=sched)

    def loss_fn(p, x, y):
        head.update(p)
        return nn.losses.cross_entropy(head(x), y, reduction="mean")

    vg = mx.value_and_grad(loss_fn)

    rng = np.random.default_rng(0)
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(len(tr))
        tot = 0.0
        for i in range(0, len(perm), batch):
            j = mx.array(perm[i:i + batch])
            loss, g = vg(params, Xtr[j], ytr[j])
            params = opt.apply_gradients(g, params)
            mx.eval(params, opt.state, loss)
            tot += loss.item() * len(j)
        head.update(params)
        ev = evaluate_affine(head, data, e)
        hist.append({"epoch": ep, "train_loss": tot / len(perm), **ev["summary"]})
        log(f"E={e} epoch {ep} loss {tot / len(perm):.3f} test top1={ev['summary']['acc']:.3f} "
            f"({time.time() - t0:.0f}s)")
    head.update(params)
    return head, {"vocab": vocab, "epochs": epochs, "lr": lr, "history": hist}


def evaluate_affine(head: AffineExit, data: Data, e: int, idx: np.ndarray | None = None,
                    chunk: int = 2048) -> dict:
    idx = data.test_idx if idx is None else idx
    H = data.h(e)
    toks, confs = [], []
    for i in range(0, len(idx), chunk):
        t, c = head.guess(to_mx(H[idx[i:i + chunk]]))
        mx.eval(t, c)
        toks.append(np.array(t))
        confs.append(np.array(c.astype(mx.float32)))
    tok = np.concatenate(toks)
    conf = np.concatenate(confs)
    acc = float((tok == data.label[idx]).mean())
    return {"tok": tok, "conf": conf, "summary": {"acc": acc,
                                                  "abstain": float((tok == -1).mean())}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6B")
    ap.add_argument("--exit", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args(argv)
    s = load(a.model)
    data = Data(a.model)
    out = data.dir / "heads"
    out.mkdir(exist_ok=True)
    for e in a.exit:
        head, info = train_affine(s, data, e, a.vocab, a.epochs, lr=a.lr)
        head.save(out / f"affine_E{e}_V{a.vocab}.npz")
        (out / f"affine_E{e}_V{a.vocab}.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
