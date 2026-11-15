"""Step 3: the layer-hot decode loop, lossless greedy.

One cycle, starting from the last accepted token y (not yet in the KV cache):
  1. Run layers [0, E) on y. The exit head guesses the next token. While the guess is
     confident and fewer than kmax rows are queued, embed the guess, run layers
     [0, E) on it (its KV entries for those layers are written now), guess again.
  2. Run layers [E, N) once over all R queued rows, reusing their layer-E states.
  3. Final norm + full LM head over the R rows (one head pass for all of them).
  4. Accept the longest prefix of guesses that equal the full model's tokens, plus
     one bonus token from the full model. Trim the KV entries of rejected rows.

With E = 1 this is the schedule in the idea note. A cycle always yields >= 1 token,
and every emitted token is the full model's greedy choice.

Usage: python -m layerhot.generate --model 0.6B --exit 2 --kmax 4 --tau 0.6 --prompt "..."
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import mlx.core as mx

from layerhot.model_utils import Split


def greedy(logits: mx.array) -> mx.array:
    """The same op mlx-lm's greedy sampler applies (argmax of log-probs)."""
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.argmax(logprobs, axis=-1)


@dataclass
class Stats:
    cycles: int = 0
    tokens: int = 0
    rows: int = 0
    guesses: int = 0
    accepted: int = 0
    head_calls: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    accept_hist: dict = field(default_factory=dict)
    guess_log: list = field(default_factory=list)  # (guess, target, accepted)

    @property
    def tokens_per_sweep(self) -> float:
        return self.tokens / max(1, self.cycles)

    @property
    def tok_s(self) -> float:
        return self.tokens / max(1e-9, self.decode_s)

    def as_dict(self) -> dict:
        return {"cycles": self.cycles, "tokens": self.tokens, "rows": self.rows,
                "guesses": self.guesses, "accepted": self.accepted,
                "head_calls": self.head_calls,
                "tokens_per_sweep": round(self.tokens_per_sweep, 4),
                "guess_precision": round(self.accepted / max(1, self.guesses), 4),
                "prefill_s": round(self.prefill_s, 4), "decode_s": round(self.decode_s, 4),
                "tok_s": round(self.tok_s, 2), "accept_hist": self.accept_hist}


class LayerHotDecoder:
    """mode="step": one host sync per guess, stop drafting at the first unconfident guess.
    mode="chain": draft all kmax-1 guesses on device (argmax feeds the next embedding),
    sync once, keep the confident prefix, trim the KV of the rest. Fewer host round trips,
    at the price of E-layer work on rows that may be dropped."""

    def __init__(self, s: Split, head, exit_layer: int, kmax: int = 4, tau: float = 0.6,
                 prefill_step: int = 2048, mode: str = "step"):
        assert 1 <= exit_layer < s.n_layers
        assert mode in ("step", "chain")
        self.s = s
        self.head = head
        self.E = exit_layer
        self.kmax = kmax
        self.tau = tau
        self.mode = mode
        self.prefill_step = prefill_step
        self.eos = set(getattr(s.tokenizer, "eos_token_ids", []) or [])

    def prefill(self, prompt: list[int]):
        s, E = self.s, self.E
        cache = s.make_cache()
        stateful = hasattr(self.head, "prefill")
        if stateful:
            self.head.reset()
        if len(prompt) < 2:
            return cache
        toks = mx.array(prompt[:-1])[None]
        hE = []
        for i in range(0, toks.shape[1], self.prefill_step):
            h = s.layers(s.embed(toks[:, i:i + self.prefill_step]), cache, 0, E)
            if stateful:
                hE.append(h)
            h = s.layers(h, cache, E, s.n_layers)
            mx.eval(h, [c.state for c in cache])
            del h
        if stateful:
            self.head.prefill(mx.concatenate(hE, axis=1) if len(hE) > 1 else hE[0])
        return cache

    def _draft_step(self, cache, y: int, stats: Stats):
        s, E = self.s, self.E
        rows = [y]
        hs = []
        h = s.layers(s.embed(mx.array([[y]])), cache, 0, E)
        hs.append(h)
        while len(rows) < self.kmax:
            tok, conf = self.head.guess(h[:, -1])
            mx.eval(tok, conf)
            stats.head_calls += 1
            g = int(tok.item())
            if g < 0 or float(conf.item()) < self.tau:
                break
            rows.append(g)
            h = s.layers(s.embed(mx.array([[g]])), cache, 0, E)
            hs.append(h)
        return rows, hs

    def _draft_chain(self, cache, y: int, stats: Stats):
        s, E = self.s, self.E
        hs, toks, confs = [], [], []
        x = mx.array([[y]])
        for i in range(self.kmax):
            h = s.layers(s.embed(x), cache, 0, E)
            hs.append(h)
            if i == self.kmax - 1:
                break
            tok, conf = self.head.guess(h[:, -1])
            toks.append(tok)
            confs.append(conf)
            x = mx.maximum(tok, 0)[:, None]  # abstain (-1) still needs a valid id to embed
        if toks:
            t, c = mx.concatenate(toks), mx.concatenate(confs)
            mx.eval(t, c, hs)
            t, c = t.tolist(), c.astype(mx.float32).tolist()
        else:
            t, c = [], []
        stats.head_calls += len(t)
        rows = [y]
        for g, p in zip(t, c):
            if g < 0 or p < self.tau:
                break
            rows.append(g)
        # drop the E-layer KV of drafted rows beyond the confident prefix
        s.trim(cache, self.kmax - len(rows), 0, E)
        return rows, hs[:len(rows)]

    def cycle(self, cache, y: int, stats: Stats, log_guesses: bool = False) -> list[int]:
        s, E = self.s, self.E
        stateful = hasattr(self.head, "settle")
        if stateful:
            self.head.mark()
        if self.mode == "chain" and self.kmax > 1:
            rows, hs = self._draft_chain(cache, y, stats)
        else:
            rows, hs = self._draft_step(cache, y, stats)
        R = len(rows)
        hE = hs[0] if R == 1 else mx.concatenate(hs, axis=1)
        out = s.layers(hE, cache, E, s.n_layers)
        target = greedy(s.head(s.norm(out)))[0]
        target = target.tolist()
        n_acc = 0
        while n_acc < R - 1 and rows[n_acc + 1] == target[n_acc]:
            n_acc += 1
        # drop KV entries of rejected rows (every layer holds R new rows now)
        s.trim(cache, R - (n_acc + 1))
        if stateful:
            self.head.settle(hs, n_acc + 1)
        stats.cycles += 1
        stats.rows += R
        stats.guesses += R - 1
        stats.accepted += n_acc
        stats.accept_hist[n_acc] = stats.accept_hist.get(n_acc, 0) + 1
        if log_guesses:
            for i in range(R - 1):
                stats.guess_log.append((rows[i + 1], target[i], i < n_acc))
        return rows[1:n_acc + 1] + [target[n_acc]]

    def generate(self, prompt: list[int], max_tokens: int = 256, log_guesses: bool = False):
        stats = Stats()
        t0 = time.perf_counter()
        cache = self.prefill(prompt)
        stats.prefill_s = time.perf_counter() - t0
        out: list[int] = []
        y = prompt[-1]
        t0 = time.perf_counter()
        while len(out) < max_tokens:
            new = self.cycle(cache, y, stats, log_guesses)
            for t in new:
                out.append(t)
                if t in self.eos or len(out) >= max_tokens:
                    break
            if out[-1] in self.eos:
                break
            y = out[-1]
        stats.decode_s = time.perf_counter() - t0
        stats.tokens = len(out)
        # an EOS is emitted by mlx-lm as a stop, not a token; mirror that
        if out and out[-1] in self.eos:
            out = out[:-1]
        return out, stats


def mlx_greedy(s: Split, prompt: list[int], max_tokens: int) -> tuple[list[int], float]:
    """Reference: mlx-lm's own greedy generate_step. Returns tokens and decode tok/s."""
    from mlx_lm.generate import generate_step

    eos = set(getattr(s.tokenizer, "eos_token_ids", []) or [])
    out = []
    t_first = None
    for i, (tok, _) in enumerate(generate_step(mx.array(prompt), s.model, max_tokens=max_tokens)):
        if i == 0:
            t_first = time.perf_counter()
        t = int(tok) if not hasattr(tok, "item") else int(tok.item())
        if t in eos:
            break
        out.append(t)
    dt = time.perf_counter() - t_first if t_first else 0.0
    # the clock starts once the first token is out, so it covers len(out) - 1 tokens
    return out, (len(out) - 1) / dt if dt > 0 else 0.0


def main(argv=None):
    from layerhot import prompts as P
    from layerhot.collect import data_dir
    from layerhot.exit_head import AffineExit, LensExit
    from layerhot.model_utils import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.6B")
    ap.add_argument("--exit", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--head", choices=["affine", "lens", "adapter"], default="affine")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--mode", choices=["step", "chain"], default="step")
    ap.add_argument("--prompt", default="Explain how a hash map works in simple terms.")
    a = ap.parse_args(argv)
    s = load(a.model)
    heads = data_dir(a.model) / "heads"
    if a.head == "affine":
        head = AffineExit.load(s, heads / f"affine_E{a.exit}_V{a.vocab}.npz").half()
    elif a.head == "adapter":
        from layerhot.adapter import AdapterExit

        head = AdapterExit.load(s, heads / f"adapter_E{a.exit}_V{a.vocab}.npz").half()
    else:
        head = LensExit(s)
    dec = LayerHotDecoder(s, head, a.exit, a.kmax, a.tau, mode=a.mode)
    ids = P.encode(s.tokenizer, {"kind": "chat", "text": a.prompt})
    ref, ref_tps = mlx_greedy(s, ids, a.max_tokens)
    out, st = dec.generate(ids, a.max_tokens)
    print(s.tokenizer.decode(out))
    print(st.as_dict())
    print("identical to mlx-lm greedy:", out == ref, f"mlx-lm {ref_tps:.1f} tok/s")


if __name__ == "__main__":
    main()
