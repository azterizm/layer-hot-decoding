"""Throttling-robust view of a head-to-head run (reads raw.json, writes analysis.json + prints markdown).

Three measures per arm and prompt kind:
  1. median decode tok/s over rounds (what the plan asks for)
  2. median over rounds of the per-round ratio to the same backend's serial arm on the same prompt
     (cancels slow drift between rounds; not position-in-round effects)
  3. tokens per target forward pass (independent of clock speed):
       ours: tokens per sweep; dflash-mlx: 1 / (1 - accepted_from_draft / generated);
       llama.cpp: predicted_n / (predicted_n - draft_n_accepted); serial arms: 1
Also: medians per session (rounds before / after a pause) to show the thermal state's effect.

Usage: python -m bench.h2h_analysis results/<ts>-head_to_head [--session-split 2]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

SERIAL = {"mlx_serial": "mlx_serial", "ours": "mlx_serial", "dflash_mlx": "mlx_serial",
          "llama_serial": "llama_serial", "llama_dflash": "llama_serial", "llama_dflash2": "llama_serial"}


def tokens_per_pass(x: dict) -> float | None:
    st = x.get("stats") or {}
    if x["arm"] in ("mlx_serial", "llama_serial"):
        return 1.0
    if x["arm"] == "ours":
        return st.get("tokens_per_sweep")
    if x["arm"] == "dflash_mlx":
        a = st.get("acceptance")
        return 1.0 / (1.0 - a) if a is not None and a < 1 else None
    n, acc = st.get("predicted_n"), st.get("draft_n_accepted")
    return n / (n - acc) if n and acc is not None and n > acc else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--session-split", type=int, default=2, help="first round of the second session")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    raw = [x for x in json.loads((d / "raw.json").read_text()) if not x["discard"]]
    kind = {p["id"]: p["kind"] for p in json.loads((d / "prompts.json").read_text())}
    arms = list(dict.fromkeys(x["arm"] for x in raw))
    by = {(x["arm"], x["round"], x["id"]): x for x in raw}
    rounds = sorted({x["round"] for x in raw})
    res = {}
    for arm in arms:
        rec = {}
        for k in ("code", "prose", "all"):
            pids = [p for p in kind if k == "all" or kind[p] == k]
            med, ratio, tpp, s1, s2 = [], [], [], [], []
            for p in pids:
                xs = [by[(arm, r, p)] for r in rounds if (arm, r, p) in by]
                if not xs:
                    continue
                med.append(statistics.median(x["tok_s"] for x in xs))
                rr = [by[(arm, r, p)]["tok_s"] / by[(SERIAL[arm], r, p)]["tok_s"]
                      for r in rounds if (arm, r, p) in by and (SERIAL[arm], r, p) in by]
                ratio.append(statistics.median(rr))
                t = [v for v in (tokens_per_pass(x) for x in xs) if v is not None]
                if t:
                    tpp.append(statistics.median(t))
                s1 += [x["tok_s"] for x in xs if x["round"] < a.session_split]
                s2 += [x["tok_s"] for x in xs if x["round"] >= a.session_split]
            rec[k] = {"median_tok_s": statistics.mean(med) if med else None,
                      "ratio_vs_serial_same_round": statistics.mean(ratio) if ratio else None,
                      "tokens_per_target_pass": statistics.mean(tpp) if tpp else None,
                      "session1_median_tok_s": statistics.median(s1) if s1 else None,
                      "session2_median_tok_s": statistics.median(s2) if s2 else None}
        res[arm] = rec
    (d / "analysis.json").write_text(json.dumps(res, indent=2))

    def f(v, n=2):
        return "–" if v is None else f"{v:.{n}f}"

    print("| arm | tok/s code | tok/s prose | tok/s all | vs own serial, same round: code | prose | all "
          "| tokens per target pass: code | prose | tok/s night (r0-1) | tok/s morning (r2-4) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, rec in res.items():
        print(f"| {arm} | {f(rec['code']['median_tok_s'], 1)} | {f(rec['prose']['median_tok_s'], 1)} | "
              f"{f(rec['all']['median_tok_s'], 1)} | {f(rec['code']['ratio_vs_serial_same_round'])} | "
              f"{f(rec['prose']['ratio_vs_serial_same_round'])} | {f(rec['all']['ratio_vs_serial_same_round'])} | "
              f"{f(rec['code']['tokens_per_target_pass'])} | {f(rec['prose']['tokens_per_target_pass'])} | "
              f"{f(rec['all']['session1_median_tok_s'], 1)} | {f(rec['all']['session2_median_tok_s'], 1)} |")


if __name__ == "__main__":
    main()
