"""Step 6.2: would the exit head add tokens on top of DFlash? (replay on DFlash's own traces)

Hybrid being priced: in each DFlash verify, after layer E of the block rows, the exit head reads
row i-1 and guesses the token at block position i. Where its guess differs from DFlash's draft
(and it is confident), a sibling row with the guess joins the verify. If DFlash's draft fails at
position f and the sibling at f holds the full model's token, the cycle commits one extra token
(the sibling, then the bonus from the sibling's row) instead of stopping at the bonus.

Phase "trace" (run with .venv-dflash): dflash-mlx greedy on held-out prompts, keeping the generated
tokens and the accepted-draft count of every cycle -> data/4B/dflash_traces.jsonl.
Phase "replay" (run with .venv): teacher-force layer E of the 4B model on prompt + output, run the
trained exit head, and count, per cycle:
  hit      = exit head confident and right at the failure position f (one extra token)
  siblings = positions i <= f where the exit head is confident and disagrees with the draft
             (for i < f the draft was right, so these rows are wasted); positions after f are on a
             wrong context and are reported with the same per-position rate as an estimate.

Usage:
  .venv-dflash/bin/python -m bench.hybrid_replay trace --n 40
  .venv/bin/python -m bench.hybrid_replay replay --exit 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRACES = REPO / "data" / "4B" / "dflash_traces.jsonl"


def trace(n: int, max_tokens: int):
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context

    from layerhot import prompts as P

    ctx = build_offline_runtime_context()
    b = load_runtime_bundle(model_ref="mlx-community/Qwen3-4B-4bit", draft_ref="z-lab/Qwen3-4B-DFlash-b16",
                            draft_quant="none", verify_config=ctx.verify)
    stop = set(get_stop_token_ids(b.tokenizer))
    test = [it for it in P.build() if it["split"] == "test"]
    pick = ([it for it in test if it["kind"] == "code"][: n * 3 // 8] + [it for it in test if it["kind"] == "chat"][: n * 3 // 8]
            + [it for it in test if it["kind"] == "prose"][: n - 2 * (n * 3 // 8)])
    with open(TRACES, "w") as f:
        for it in pick:
            ids = P.encode(b.tokenizer, it)
            stream = stream_dflash_generate(
                target_model=b.target_model, target_ops=b.target_ops, tokenizer=b.tokenizer,
                draft_model=b.draft_model, draft_backend=b.draft_backend, prompt="", max_new_tokens=max_tokens,
                use_chat_template=False, stop_token_ids=list(stop), prompt_tokens_override=list(ids),
                runtime_context=ctx)
            toks, summ = [], None
            try:
                for ev in stream:
                    if isinstance(ev, TokenEvent):
                        toks.append(int(ev.token_id))
                    elif isinstance(ev, SummaryEvent):
                        summ = ev
            finally:
                stream.close()
            rec = {"id": it["id"], "kind": it["kind"], "prompt_ids": ids, "gen_ids": toks,
                   "acceptance_history": list(summ.acceptance_history), "block_tokens": summ.block_tokens,
                   "acceptance_ratio": float(summ.acceptance_ratio), "copyspec_hits": int(summ.copyspec_hits)}
            f.write(json.dumps(rec) + "\n")
            print(it["id"], it["kind"], len(toks), "cycles", len(summ.acceptance_history),
                  "tokens/cycle", round(len(toks) / max(1, len(summ.acceptance_history)), 2), flush=True)


def replay(exit_layer: int, taus: list[float]):
    import mlx.core as mx
    import numpy as np

    from bench import snapshot
    from layerhot.collect import data_dir
    from layerhot.exit_head import AffineExit
    from layerhot.model_utils import load

    s = load("4B")
    head = AffineExit.load(s, data_dir("4B") / "heads" / f"affine_E{exit_layer}_V16384.npz").half()
    recs = [json.loads(l) for l in TRACES.read_text().splitlines()]
    out = snapshot.new_results_dir(f"hybrid_replay_E{exit_layer}")
    snapshot.write_env(out, {"short": snapshot.short_snapshot(), "software": snapshot.software(),
                             "models": [snapshot.model_info("mlx-community/Qwen3-4B-4bit")]})
    per_tau = {t: {"cycles": 0, "fail_cycles": 0, "hits": 0, "siblings_le_f": 0, "positions_le_f": 0,
                   "tokens": 0, "by_kind": {}} for t in taus}
    check = []
    for r in recs:
        ids, gen = r["prompt_ids"], r["gen_ids"]
        seq = ids + gen
        h = s.layers(s.embed(mx.array([seq[:-1]])), None, 0, exit_layer)
        tok, conf = head.guess(h[0])
        mx.eval(tok, conf)
        tok, conf = np.array(tok), np.array(conf.astype(mx.float32))
        # guess for absolute position q (the token seq[q]) comes from the state at q-1
        P0 = len(ids)
        hist = r["acceptance_history"]
        B = r["block_tokens"] or 16
        # gen[0] comes from the prefill; cycle c starts after the last committed token
        pos = 1  # index into gen of the first token committed by cycle 0
        cyc = []
        for a in hist:
            if pos >= len(gen):
                break
            f = pos + a  # gen index of the bonus token = first position where the draft failed (if a < B-1)
            cyc.append((pos, a, f))
            pos = f + 1
        check.append((len(gen), 1 + sum(a + 1 for _, a, _ in cyc)))
        for t in taus:
            st = per_tau[t]
            k = st["by_kind"].setdefault(r["kind"], {"cycles": 0, "hits": 0, "tokens": 0})
            for start, a, f in cyc:
                if f >= len(gen):
                    continue
                st["cycles"] += 1
                k["cycles"] += 1
                st["tokens"] += a + 1
                k["tokens"] += a + 1
                full_block = a >= B - 1
                if not full_block:
                    st["fail_cycles"] += 1
                    q = P0 + f  # absolute position of the bonus (= full model's token where the draft failed)
                    g, c = int(tok[q - 1]), float(conf[q - 1])
                    if g >= 0 and c >= t and g == seq[q]:
                        st["hits"] += 1
                        k["hits"] += 1
                # wasted siblings at accepted positions start..f-1: exit head disagrees with a correct draft
                for i in range(start, f):
                    q = P0 + i
                    g, c = int(tok[q - 1]), float(conf[q - 1])
                    st["positions_le_f"] += 1
                    if g >= 0 and c >= t and g != seq[q]:
                        st["siblings_le_f"] += 1
    res = {"exit": exit_layer, "n_sequences": len(recs), "count_check_gen_vs_cycles": check[:10], "taus": {}}
    for t, st in per_tau.items():
        tpc = st["tokens"] / st["cycles"]
        hit_rate = st["hits"] / st["cycles"]
        waste_rate = st["siblings_le_f"] / max(1, st["positions_le_f"])
        res["taus"][t] = {
            "dflash_tokens_per_cycle": tpc,
            "share_cycles_ending_in_rejection": st["fail_cycles"] / st["cycles"],
            "extra_tokens_per_cycle": hit_rate,
            "hit_rate_at_failure": st["hits"] / max(1, st["fail_cycles"]),
            "relative_token_gain": hit_rate / tpc,
            "wasted_sibling_rate_per_accepted_position": waste_rate,
            "est_sibling_rows_per_cycle": hit_rate + waste_rate * (B - 1),
            "by_kind": {k: {"extra_tokens_per_cycle": v["hits"] / max(1, v["cycles"]),
                            "relative_token_gain": (v["hits"] / max(1, v["cycles"])) / (v["tokens"] / max(1, v["cycles"]))}
                        for k, v in st["by_kind"].items()},
        }
    (out / "hybrid_replay.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("results:", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["trace", "replay"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--exit", type=int, default=1)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    a = ap.parse_args(argv)
    if a.phase == "trace":
        trace(a.n, a.max_tokens)
    else:
        replay(a.exit, a.taus)


if __name__ == "__main__":
    sys.exit(main())
