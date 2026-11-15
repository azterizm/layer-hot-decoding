"""Step 5: head-to-head on this M4, Qwen3-4B, T=0.

Each round runs every arm once as its own process (load, warm-up, one generation per
prompt), arms rotated between rounds, 60 s cool-down, machine snapshot around each
round. Speed = decode tok/s (prefill excluded), median over kept rounds.

Correctness gate: a speculative arm's speed counts only for prompts where its tokens equal
its own backend's serial tokens (MLX arms vs mlx_serial, llama arms vs llama_serial).
Mismatches are reported with the first diverging position.

Usage: python -m bench.head_to_head --arms mlx_serial ours dflash_mlx llama_serial llama_dflash2
         --ours '{"head":"affine","exit":2,"kmax":2,"tau":0.5}' [--rounds 5] [--max-tokens 256]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from bench import snapshot

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / ".venv" / "bin" / "python")
PY_DFLASH = str(REPO / ".venv-dflash" / "bin" / "python")
GGUF_DIR = Path.home() / "models" / "gguf"

DRAFTS = {
    "dflash_mlx": "z-lab/Qwen3-4B-DFlash-b16",
    "llama_dflash": str(GGUF_DIR / "Qwen3-4B-DFlash-b16.gguf"),
    "llama_dflash2": str(GGUF_DIR / "Qwen3-4B-DFlash2.gguf"),
}
REF = {"ours": "mlx_serial", "dflash_mlx": "mlx_serial",
       "llama_dflash": "llama_serial", "llama_dflash2": "llama_serial"}


def build_prompts(n_code: int, n_prose: int, mlx_model: str) -> list[dict]:
    from mlx_lm.utils import load_tokenizer
    from mlx_lm.utils import _download as dl  # noqa: PLC2701

    from layerhot import prompts as P

    tok = load_tokenizer(Path(dl(mlx_model)))
    test = [it for it in P.build() if it["split"] == "test"]
    code = [it for it in test if it["kind"] == "code"][:n_code]
    prose = [it for it in test if it["kind"] == "chat"][:n_prose]
    return [{"id": it["id"], "kind": "code" if it["kind"] == "code" else "prose",
             "text": it["text"], "ids": P.encode(tok, it)} for it in code + prose]


def first_diff(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["mlx_serial", "ours", "dflash_mlx", "llama_serial", "llama_dflash", "llama_dflash2"])
    ap.add_argument("--ours", default='{"head":"affine","exit":2,"kmax":2,"tau":0.5}')
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n-code", type=int, default=3)
    ap.add_argument("--n-prose", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--mlx-model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--tag", default="head_to_head")
    ap.add_argument("--resume", default=None,
                    help="existing results dir: keep its prompts and finished rounds, run the rest")
    a = ap.parse_args(argv)

    if a.resume:
        out = Path(a.resume)
        prompts = json.loads((out / "prompts.json").read_text())
        (out / f"env_resume_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps(snapshot.full_snapshot(60), indent=2))
    else:
        out = snapshot.new_results_dir(a.tag)
        snapshot.write_env(out, snapshot.full_snapshot(60, [a.mlx_model, str(GGUF_DIR / "Qwen3-4B-Q4_0.gguf")]
                                                       + [d for d in DRAFTS.values()]))
        prompts = build_prompts(a.n_code, a.n_prose, a.mlx_model)
        (out / "prompts.json").write_text(json.dumps(prompts, indent=1))
    pfile = out / "prompts.json"
    print("results:", out, flush=True)

    ran_one = False
    for r in range(a.rounds):
        if (out / f"round_{r}_env.json").exists():
            print(f"round {r}: already done, kept", flush=True)
            continue
        if ran_one:
            time.sleep(a.cooldown)
        ran_one = True
        snapshot.wait_quiet()
        before = snapshot.short_snapshot()
        order = a.arms[r % len(a.arms):] + a.arms[:r % len(a.arms)]
        for arm in order:
            of = out / f"r{r}_{arm}.json"
            py = PY_DFLASH if arm == "dflash_mlx" else PY
            cmd = [py, "-m", "bench.h2h_arm", "--arm", arm, "--prompts", str(pfile), "--out", str(of),
                   "--max-tokens", str(a.max_tokens), "--mlx-model", a.mlx_model, "--ours", a.ours]
            if arm in DRAFTS:
                cmd += ["--draft", DRAFTS[arm]]
            p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                               env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin:/opt/homebrew/bin",
                                    "HOME": str(Path.home()), "LLAMA_LOG": str(out / f"r{r}_{arm}.log")})
            if p.returncode != 0:
                print(f"round {r} {arm} FAILED:\n{p.stderr[-2000:]}", flush=True)
                continue
            res = json.loads(of.read_text())
            print(f"round {r} {arm}: " + " ".join(f"{x['tok_s']:.1f}" for x in res["results"]), flush=True)
        after = snapshot.short_snapshot()
        reasons = snapshot.discard_reasons(before, after)
        (out / f"round_{r}_env.json").write_text(json.dumps({"before": before, "after": after,
                                                             "discard": reasons}, indent=2))
        print(f"round {r} discard={reasons}", flush=True)

    # rebuild every run from the per-round files (also covers rounds kept from a resumed dir)
    runs: list[dict] = []
    for r in range(a.rounds):
        ef = out / f"round_{r}_env.json"
        if not ef.exists():
            continue
        reasons = json.loads(ef.read_text())["discard"]
        for arm in a.arms:
            of = out / f"r{r}_{arm}.json"
            if of.exists():
                for x in json.loads(of.read_text())["results"]:
                    runs.append({"round": r, "arm": arm, "discard": reasons, **x})
    (out / "raw.json").write_text(json.dumps(runs))
    kept = [x for x in runs if not x["discard"]] or runs
    kind = {p["id"]: p["kind"] for p in prompts}
    # reference tokens per backend (round 0 serial; serial is deterministic)
    ref_tok = {}
    for x in runs:
        if x["arm"] in ("mlx_serial", "llama_serial"):
            ref_tok.setdefault((x["arm"], x["id"]), x["tokens"])
    summary = {"arms": {}, "all_rounds_discarded": all(x["discard"] for x in runs), "prompts": len(prompts)}
    for arm in a.arms:
        xs = [x for x in kept if x["arm"] == arm]
        if not xs:
            continue
        per_prompt = {}
        mism = []
        for pid in sorted({x["id"] for x in xs}):
            ys = [x for x in xs if x["id"] == pid]
            ok = True
            if arm in REF and (REF[arm], pid) in ref_tok:
                d = first_diff(ys[0]["tokens"], ref_tok[(REF[arm], pid)])
                if d is not None:
                    ok = False
                    mism.append({"id": pid, "first_diff": d, "len": len(ys[0]["tokens"])})
            per_prompt[pid] = {"median_tok_s": statistics.median(y["tok_s"] for y in ys),
                               "identical_to_serial": ok, "n_tokens": len(ys[0]["tokens"]),
                               "stats": ys[0].get("stats")}
        rec = {"per_prompt": per_prompt, "mismatches": mism}
        for k in ("code", "prose", "all"):
            sel = [v["median_tok_s"] for pid, v in per_prompt.items()
                   if (k == "all" or kind[pid] == k)]
            sel_ok = [v["median_tok_s"] for pid, v in per_prompt.items()
                      if (k == "all" or kind[pid] == k) and v["identical_to_serial"]]
            rec[f"{k}_tok_s"] = statistics.mean(sel) if sel else None
            rec[f"{k}_tok_s_identical_only"] = statistics.mean(sel_ok) if sel_ok else None
        summary["arms"][arm] = rec
    for arm, rec in summary["arms"].items():
        for base in ("mlx_serial", "llama_serial"):
            if base in summary["arms"]:
                b = summary["arms"][base]
                rec[f"speedup_vs_{base}"] = {k: (rec[f"{k}_tok_s"] / b[f"{k}_tok_s"]) if b[f"{k}_tok_s"] else None
                                             for k in ("code", "prose", "all")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for arm, rec in summary["arms"].items():
        print(f"{arm:14s} code {rec['code_tok_s']:.1f}  prose {rec['prose_tok_s']:.1f}  all {rec['all_tok_s']:.1f} tok/s"
              f"  mismatches={len(rec['mismatches'])}  "
              + " ".join(f"x{k[11:]}={v['all']:.2f}" for k, v in rec.items() if k.startswith("speedup_vs_") and v["all"]))


if __name__ == "__main__":
    sys.exit(main())
