"""Print the RESULTS.md tables from results/*/ JSON files (no hand-copied numbers).

Usage: python -m bench.report [--mlx DIR ...] [--llama DIR] [--acceptance DIR ...] [--tree DIR ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def mlx_table(dirs: list[str]) -> str:
    rows = {}
    parts = {}
    for d in dirs:
        s = json.loads((Path(d) / "summary.json").read_text())
        for r in s["c_of_K"]:
            rows.setdefault((r["model"], r["ctx"]), {})[r["K"]] = r
        for p in s["parts"]:
            parts.setdefault((p["model"], p["ctx"]), {})[p["what"]] = p
    out = ["| model | ctx | sweep ms (K=1) | c(2) | c(3) | c(4) | c(8) | LM head share | 1 layer, 1 row (ms) | 16k head (ms) | source |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {"0.6B": 0, "1.7B": 1, "4B": 2}
    for (m, ctx), ks in sorted(rows.items(), key=lambda x: (order.get(x[0][0], 9), x[0][1])):
        p = parts.get((m, ctx), {})
        src = next(d for d in dirs if any(r["model"] == m for r in json.loads((Path(d) / "summary.json").read_text())["c_of_K"]))
        out.append(f"| {m} | {ctx} | {fmt(ks[1]['sweep_ms'], 1)} | " + " | ".join(fmt(ks[k]['c(K)']) for k in (2, 3, 4, 8))
                   + f" | {ks[1]['head_share'] * 100:.0f}% | {fmt(p.get('layers_E1', {}).get('ms'))} | {fmt(p.get('head_16k', {}).get('ms'))} | `{Path(src).name}` |")
    return "\n".join(out)


def llama_table(d: str) -> str:
    s = json.loads((Path(d) / "summary.json").read_text())
    rows = {}
    for r in s["c_of_K"]:
        rows.setdefault((r["model"], r["backend"], r["ctx"]), {})[r["K"]] = r
    out = ["| model | backend | ctx | step ms (K=1) | serial tok/s | c(2) | c(3) | c(4) | c(8) |",
           "|---|---|---|---|---|---|---|---|---|"]
    order = {"0.6B": 0, "1.7B": 1, "4B": 2}
    for (m, b, ctx), ks in sorted(rows.items(), key=lambda x: (order.get(x[0][0], 9), x[0][1], x[0][2])):
        out.append(f"| {m} | {b} | {ctx} | {fmt(ks[1]['step_ms'], 1)} | {fmt(ks[1]['serial_tok_s'], 1)} | "
                   + " | ".join(fmt(ks[k]['c(K)']) for k in (2, 3, 4, 8)) + " |")
    tg = s.get("llama_bench_tg", {})
    if tg:
        out.append("")
        out.append("llama-bench tg128 at depth 512 (cross-check of the K=1 column): " + ", ".join(
            f"{k} {v.get('avg_ts', 0):.1f} tok/s" + (" (discarded round)" if v.get("discard") else "") for k, v in tg.items()))
    return "\n".join(out)


def acceptance_tables(dirs: list[str]) -> str:
    out = []
    for d in dirs:
        r = json.loads((Path(d) / "acceptance.json").read_text())
        out.append(f"**{r['model']}** (`{Path(d).name}`): {r['train_positions']} train / {r['test_positions']} held-out "
                   f"positions. Sanity: logit lens at the last layer = full model "
                   f"{r['sanity']['lens_at_N_eq_label'] * 100:.1f}%; teacher-forced label = generated token "
                   f"{r['sanity']['label_eq_generated'] * 100:.1f}%. Next-token classes: "
                   + ", ".join(f"{k} {v * 100:.0f}%" for k, v in r["class_share"].items()) + ".")
        out.append("")
        out.append("| E | head | top-1 | punct/ws | function | subword | content | precision @ coverage (τ=0.5) | best replay (kmax, τ) | tokens/sweep | cost/sweep | predicted speedup | same, no host overhead |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for e, rec in r["exits"].items():
            for h in ("lens", "affine", "adapter"):
                if h not in rec:
                    continue
                a = rec[h]
                pr = next(p for p in a["pr_curve"] if p["tau"] == 0.5)
                b = max(a["replay"], key=lambda x: x["pred_speedup"])
                bi = max(a["replay"], key=lambda x: x.get("pred_speedup_no_host_overhead", 0))
                c = a["by_class"]
                out.append(f"| {e} | {h} | {a['top1'] * 100:.1f}% | {c.get('punct/ws', 0) * 100:.0f}% | {c.get('function', 0) * 100:.0f}% | "
                           f"{c.get('subword', 0) * 100:.0f}% | {c.get('content', 0) * 100:.0f}% | "
                           f"{fmt((pr['precision'] or 0) * 100, 0)}% @ {pr['coverage'] * 100:.0f}% | ({b['kmax']}, {b['tau']}) | "
                           f"{b['tokens_per_sweep']:.3f} | {b['cost_per_sweep']:.3f} | **{b['pred_speedup']:.3f}** | "
                           f"{fmt(bi.get('pred_speedup_no_host_overhead'), 3)} |")
        out.append("")
        out.append(f"Best: E={r['best']['E']} {r['best']['head']}, predicted {r['best']['pred_speedup']:.3f}x "
                   f"→ gate: **{r['gate']}**.")
        out.append("")
    return "\n".join(out)


def tree_table(dirs: list[str]) -> str:
    out = ["| model | E | top-1 | top-2 | best predicted speedup | share of cycles using the 2-branch tree |",
           "|---|---|---|---|---|---|"]
    for d in dirs:
        r = json.loads((Path(d) / "tree_top2.json").read_text())
        out.append(f"| {r['model']} | {r['exit']} | {r['top1'] * 100:.1f}% | {r['top2'] * 100:.1f}% | "
                   f"{r['best']['pred_speedup']:.3f} | {r['best']['share_tree'] * 100:.0f}% |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx", nargs="*", default=[])
    ap.add_argument("--llama", default=None)
    ap.add_argument("--acceptance", nargs="*", default=[])
    ap.add_argument("--tree", nargs="*", default=[])
    a = ap.parse_args(argv)
    if a.mlx:
        print(mlx_table(a.mlx), "\n")
    if a.llama:
        print(llama_table(a.llama), "\n")
    if a.acceptance:
        print(acceptance_tables(a.acceptance))
    if a.tree:
        print(tree_table(a.tree))


if __name__ == "__main__":
    main()
