"""Step 4a: Kangaroo-style adapter exit head (model frozen, decoding still lossless).

h_E -> a = h_E + Attn(RMSNorm(h_E))   one causal self-attention block, own KV cache
    -> A a + b -> final norm (frozen) -> reduced-vocab head (frozen) + abstain logit

The attention block starts as a dequantized copy of the model's own next attention
layer (layer E), so it begins as "what the model does next" and is then trained.
Loss: cross-entropy on the full model's greedy token + KL to the full model's
distribution (over the reduced vocabulary, with the remaining mass on abstain).
The layer-E states of the whole sequence (prompt included) are recomputed on the fly,
so the adapter attends over the same context it sees when decoding.

Train: python -m layerhot.adapter --model 0.6B --exit 2
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
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import Attention

from layerhot.collect import data_dir
from layerhot.exit_head import Data, map_labels, to_mx
from layerhot.model_utils import ReducedHead, load


def _dequant_linear(q) -> nn.Linear:
    if isinstance(q, nn.QuantizedLinear):
        w = mx.dequantize(q.weight, q.scales, q.biases, q.group_size, q.bits,
                          mode=getattr(q, "mode", "affine"))
    else:
        w = q.weight
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False)
    lin.weight = w.astype(mx.float32)
    return lin


class AdapterExit(nn.Module):
    def __init__(self, s, rows: np.ndarray, init_layer: int):
        super().__init__()
        H = s.hidden
        src = s.inner.layers[init_layer]
        self.norm_in = nn.RMSNorm(H, eps=s.model.args.rms_norm_eps)
        self.norm_in.weight = src.input_layernorm.weight.astype(mx.float32)
        self.attn = Attention(s.model.args)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.attn, name, _dequant_linear(getattr(src.self_attn, name)))
        self.attn.q_norm.weight = src.self_attn.q_norm.weight.astype(mx.float32)
        self.attn.k_norm.weight = src.self_attn.k_norm.weight.astype(mx.float32)
        self.A = mx.eye(H, dtype=mx.float32)
        self.b = mx.zeros((H,), dtype=mx.float32)
        self.w_abs = mx.zeros((H,), dtype=mx.float32)
        self.b_abs = mx.array([-2.0], dtype=mx.float32)
        self.vocab_rows = mx.array(rows.astype(np.int32))
        self.freeze(keys=["vocab_rows"])
        self._final_norm = s.inner.norm
        self._head = ReducedHead.from_model(s, self.vocab_rows)
        self._cache: KVCache | None = None
        self._mark = 0
        self.init_layer = init_layer

    # -------------------------------------------------- shared forward
    def logits(self, h: mx.array, mask=None, cache=None) -> mx.array:
        dt = self.A.dtype
        x = h.astype(dt)
        a = x + self.attn(self.norm_in(x), mask, cache)
        y = a @ self.A.T + self.b
        n = self._final_norm(y.astype(mx.bfloat16))
        lv = self._head(n).astype(mx.float32)
        la = (n.astype(mx.float32) @ self.w_abs.astype(mx.float32))[..., None] + self.b_abs
        return mx.concatenate([lv, la], axis=-1)

    def _decode(self, logits: mx.array):
        p = mx.softmax(logits, axis=-1)
        idx = mx.argmax(p, axis=-1)
        conf = mx.max(p, axis=-1)
        V = self.vocab_rows.shape[0]
        tok = mx.where(idx == V, -1, self.vocab_rows[mx.minimum(idx, V - 1)])
        return tok, conf

    # -------------------------------------------------- decoding (stateful)
    def reset(self):
        self._cache = KVCache()

    def prefill(self, hE: mx.array):
        """hE: (1, T, H) layer-E states of the prompt prefix."""
        self.reset()
        self.logits(hE, mask="causal" if hE.shape[1] > 1 else None, cache=self._cache)
        mx.eval(self._cache.state)

    def mark(self):
        self._mark = self._cache.offset

    def guess(self, h: mx.array):
        """h: (1, H) state of the newest row. Appends that row to the adapter cache."""
        lg = self.logits(h[:, None, :], None, self._cache)[:, -1]
        return self._decode(lg)

    def settle(self, hs: list[mx.array], keep: int):
        """After verify: leave exactly `keep` rows of this cycle in the adapter cache."""
        present = self._cache.offset - self._mark
        if present > keep:
            self._cache.trim(present - keep)
        elif present < keep:
            h = mx.concatenate(hs[present:keep], axis=1)
            self.logits(h, "causal" if h.shape[1] > 1 else None, self._cache)

    def half(self):
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(self.attn, name)
            lin.weight = lin.weight.astype(mx.bfloat16)
        for m in (self.norm_in, self.attn.q_norm, self.attn.k_norm):
            m.weight = m.weight.astype(mx.bfloat16)
        self.A = self.A.astype(mx.bfloat16)
        self.b = self.b.astype(mx.bfloat16)
        return self

    def trainable(self):
        return self.trainable_parameters()

    def save(self, path: Path):
        mx.savez(str(path), **{k: v for k, v in tree_flatten(self.parameters())},
                 init_layer=mx.array(self.init_layer))

    @classmethod
    def load(cls, s, path: Path) -> "AdapterExit":
        w = dict(mx.load(str(path)))
        m = cls(s, np.array(w.pop("vocab_rows")), int(w.pop("init_layer").item()))
        m.load_weights(list(w.items()), strict=False)
        return m


# ------------------------------------------------------------------ training

def sequences(data: Data) -> list[dict]:
    gen = [json.loads(l) for l in (data.dir / "gen.jsonl").read_text().splitlines()]
    off = 0
    split_of = {int(p): bool(t) for p, t in zip(data.meta["prompt"], data.is_test)}
    out = []
    for g in gen:
        n = len(g["gen_ids"])
        if n:
            out.append({"id": g["id"], "ids": g["prompt_ids"] + g["gen_ids"], "P": len(g["prompt_ids"]),
                        "off": off, "n": n, "test": split_of.get(g["id"], False)})
        off += n
    return out


def train_adapter(s, data: Data, e: int, vocab: int = 16384, epochs: int = 6, lr: float = 3e-4,
                  kl_weight: float = 1.0, log=print):
    rows = data.vocab(vocab)
    ad = AdapterExit(s, rows, init_layer=e)
    y_all = map_labels(data.label, rows)
    hN = data.h(data.info["n_layers"])
    seqs = sequences(data)
    train = [q for q in seqs if not q["test"]]
    rows_mx = mx.array(rows.astype(np.int32))

    def target_dist(q):
        hn = to_mx(hN[q["off"]:q["off"] + q["n"]])
        lp = nn.log_softmax(s.head(s.norm(hn)).astype(mx.float32), axis=-1)
        p_in = mx.exp(lp[:, rows_mx])
        p_abs = mx.maximum(1.0 - p_in.sum(-1, keepdims=True), 1e-8)
        return mx.concatenate([p_in, p_abs], axis=-1)

    params = ad.trainable()
    steps = epochs * len(train)
    opt = optim.Adam(learning_rate=optim.cosine_decay(lr, steps))
    rng = np.random.default_rng(0)
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        tot = 0.0
        for qi in rng.permutation(len(train)):
            q = train[qi]
            ids = mx.array([q["ids"][:-1]])
            h = mx.stop_gradient(s.layers(s.embed(ids), None, 0, e))
            # loss positions P-1 .. T-2 of the sequence = rows P-1.. of h
            P = q["P"]
            y = mx.array(y_all[q["off"]:q["off"] + q["n"]])
            p_t = mx.stop_gradient(target_dist(q))

            def lf(p, h=h, y=y, p_t=p_t, P=P):
                ad.update(p)
                lg = ad.logits(h, mask="causal")[0, P - 1:]
                ce = nn.losses.cross_entropy(lg, y, reduction="mean")
                lq = nn.log_softmax(lg, axis=-1)
                kl = (p_t * (mx.log(p_t + 1e-9) - lq)).sum(-1).mean()
                return ce + kl_weight * kl

            loss, g = mx.value_and_grad(lf)(params)
            params = opt.apply_gradients(g, params)
            mx.eval(params, opt.state, loss)
            tot += loss.item()
        ad.update(params)
        ev = evaluate_adapter(s, ad, data, e)
        hist.append({"epoch": ep, "train_loss": tot / len(train), "acc": ev["acc"]})
        log(f"adapter E={e} epoch {ep} loss {tot / len(train):.3f} test top1={ev['acc']:.3f} "
            f"({time.time() - t0:.0f}s)")
    ad.update(params)
    return ad, {"vocab": vocab, "epochs": epochs, "lr": lr, "kl_weight": kl_weight, "history": hist}


def evaluate_adapter(s, ad: AdapterExit, data: Data, e: int) -> dict:
    """Guesses for every held-out position, aligned to data.test_idx order."""
    seqs = [q for q in sequences(data) if q["test"]]
    tok = np.full(len(data.label), -2, dtype=np.int64)
    conf = np.zeros(len(data.label), dtype=np.float32)
    for q in seqs:
        h = s.layers(s.embed(mx.array([q["ids"][:-1]])), None, 0, e)
        lg = ad.logits(h, mask="causal")[0, q["P"] - 1:]
        t, c = ad._decode(lg)
        mx.eval(t, c)
        tok[q["off"]:q["off"] + q["n"]] = np.array(t)
        conf[q["off"]:q["off"] + q["n"]] = np.array(c)
    te = data.test_idx
    assert (tok[te] != -2).all()
    return {"tok": tok[te], "conf": conf[te], "acc": float((tok[te] == data.label[te]).mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6B")
    ap.add_argument("--exit", type=int, nargs="+", default=[2])
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args(argv)
    s = load(a.model)
    data = Data(a.model)
    out = data.dir / "heads"
    out.mkdir(exist_ok=True)
    for e in a.exit:
        ad, info = train_adapter(s, data, e, a.vocab, a.epochs, a.lr)
        ad.save(out / f"adapter_E{e}_V{a.vocab}.npz")
        (out / f"adapter_E{e}_V{a.vocab}.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
