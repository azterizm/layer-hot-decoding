"""Step 6.3: the layer-hot loop without host round trips, pipelined like mlx-lm.

Same schedule as `LayerHotDecoder` with kmax = 2 and no confidence gate (the guess row always
rides the verify), but every decision stays on the GPU:

  - DeviceKVCache keeps the write position as a device scalar; RoPE takes it as an array offset,
    rows are written with a dynamic slice update, and a boolean mask hides slots beyond the
    committed length. A rejected guess row is never trimmed: the next cycle writes over it.
  - The cycle's outputs (next token, new committed length) are lazy arrays that feed the next
    cycle's graph directly, so cycle c+1 is built and queued (`mx.async_eval`) before the host
    reads cycle c's tokens, the same overlap mlx-lm's `generate_step` gets.

The host keeps only an upper bound on the committed length (+2 per cycle), used to size the
key slice; the mask makes the extra slots inert.
"""

from __future__ import annotations

import time

import mlx.core as mx

from layerhot.generate import Stats, greedy
from layerhot.model_utils import Split


class DeviceKVCache:
    def __init__(self, max_len: int, n_kv: int, head_dim: int, dtype=mx.bfloat16):
        self.keys = mx.zeros((1, n_kv, max_len, head_dim), dtype)
        self.values = mx.zeros((1, n_kv, max_len, head_dim), dtype)
        self.offset = mx.array(0, dtype=mx.int32)  # write position of the next update (device)
        self.limit = 0  # python: key slots exposed to attention after the update

    def update_and_fetch(self, k, v):
        start = self.offset.reshape(1)
        self.keys = mx.slice_update(self.keys, k.astype(self.keys.dtype), start, axes=(2,))
        self.values = mx.slice_update(self.values, v.astype(self.values.dtype), start, axes=(2,))
        return self.keys[..., : self.limit, :], self.values[..., : self.limit, :]

    def make_mask(self, N, return_array=False, window_size=None):
        j = mx.arange(self.limit)[None, :]
        i = mx.arange(N)[:, None]
        return j <= (self.offset + i)

    # mlx-lm helpers look for these
    @property
    def state(self):
        return self.keys, self.values

    def empty(self):
        return False


class PipelinedLayerHotDecoder:
    def __init__(self, s: Split, head, exit_layer: int, max_len: int = 4096, guess: bool = True):
        """guess=False runs plain one-row decoding through the same device-side cache and
        pipelining (a diagnostic baseline: isolates this machinery's cost from the guess row)."""
        self.s, self.head, self.E = s, head, exit_layer
        self.guess = guess
        a = s.model.args
        self.n_kv, self.head_dim = a.num_key_value_heads, a.head_dim
        self.max_len = max_len
        self.eos = set(getattr(s.tokenizer, "eos_token_ids", []) or [])

    def _set(self, caches, start, end, offset, limit):
        for c in caches[start:end]:
            c.offset = offset
            c.limit = limit

    def _cycle(self, caches, y, base, ub):
        """y: (1,) last committed token (not yet in the cache); base: () committed length;
        ub: python upper bound on base. Returns lazy (t (2,), acc (), y_next (1,), base_next ())."""
        s, E, N = self.s, self.E, self.s.n_layers
        if not self.guess:
            self._set(caches, 0, N, base, ub + 1)
            out = s.layers(s.embed(y[None]), caches, 0, N)
            t = greedy(s.head(s.norm(out)))[0].astype(mx.int32)  # (1,)
            t = mx.concatenate([t, t])
            return t, mx.array(False), t[:1], base + 1
        self._set(caches, 0, E, base, ub + 1)
        h0 = s.layers(s.embed(y[None]), caches, 0, E)
        g, _ = self.head.guess(h0[:, -1])
        g = mx.maximum(g, 0).astype(mx.int32)  # abstain (-1) still needs a valid id; it can never match
        self._set(caches, 0, E, base + 1, ub + 2)
        h1 = s.layers(s.embed(g[None]), caches, 0, E)
        self._set(caches, E, N, base, ub + 2)
        out = s.layers(mx.concatenate([h0, h1], axis=1), caches, E, N)
        t = greedy(s.head(s.norm(out)))[0].astype(mx.int32)  # (2,)
        acc = g[0] == t[0]
        y_next = mx.where(acc, t[1], t[0]).reshape(1)
        base_next = base + 1 + acc.astype(mx.int32)
        return t, acc, y_next, base_next

    def generate(self, prompt: list[int], max_tokens: int = 256, log_guesses: bool = False):
        s = self.s
        stats = Stats()
        P = len(prompt)
        max_len = min(self.max_len, P + 2 * (max_tokens + 2) + 8)
        caches = [DeviceKVCache(max_len, self.n_kv, self.head_dim) for _ in range(s.n_layers)]
        t0 = time.perf_counter()
        if P > 1:
            self._set(caches, 0, s.n_layers, mx.array(0, dtype=mx.int32), P - 1)
            h = s.layers(s.embed(mx.array([prompt[:-1]])), caches, 0, s.n_layers)
            mx.eval(h, [c.keys for c in caches], [c.values for c in caches])
        stats.prefill_s = time.perf_counter() - t0

        out: list[int] = []
        y = mx.array([prompt[-1]], dtype=mx.int32)
        base = mx.array(P - 1, dtype=mx.int32)
        known = P - 1  # exact committed length at the start of `cur` (the host learns it one cycle late)
        t0 = time.perf_counter()
        cur = self._cycle(caches, y, base, known)
        mx.async_eval(*cur)
        done = False
        step = 2 if self.guess else 1  # most slots a cycle can commit
        while not done:
            # `nxt` starts at known + (1 or step): bound the key slice by that, not by a running sum,
            # so the masked tail stays <= 2 slots instead of growing by (step - accepted) per cycle
            nxt = self._cycle(caches, cur[2], cur[3], known + step)  # queued before reading `cur`
            mx.async_eval(*nxt)
            t, acc = cur[0].tolist(), bool(cur[1].item())
            known += 1 + int(acc and self.guess)
            stats.cycles += 1
            stats.rows += step
            stats.guesses += step - 1
            stats.accepted += int(acc)
            stats.accept_hist[int(acc)] = stats.accept_hist.get(int(acc), 0) + 1
            for tok in (t[0], t[1]) if acc else (t[0],):
                out.append(tok)
                if tok in self.eos or len(out) >= max_tokens:
                    done = True
                    break
            cur = nxt
        mx.eval(*cur)  # let the speculative extra cycle finish before timing stops
        stats.decode_s = time.perf_counter() - t0
        stats.tokens = len(out)
        if out and out[-1] in self.eos:
            out = out[:-1]
        return out, stats
