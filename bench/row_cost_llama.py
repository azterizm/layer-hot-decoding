"""Step 1 (llama.cpp): c(K) from llama-batched-bench, Metal and CPU.

K parallel sequences share one prompt (-pps), so each decode step pushes K rows
through every weight matrix against roughly the same KV context, which is the
cost the layer-hot idea depends on. c(K) = step time at K / step time at 1.

Also runs llama-bench tg as a cross-check on the serial (K=1) speed.

Usage: python -m bench.row_cost_llama --models 0.6B 1.7B 4B [--rounds 3] [--cpu-model 0.6B]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from bench import snapshot

BIN = snapshot.LLAMA_CPP / "build" / "bin"
GGUF = Path.home() / "models" / "gguf"
NPL = [1, 2, 3, 4, 8]
NPP = [512, 2048]
NTG = 128


def gguf(name: str) -> Path:
    return GGUF / f"Qwen3-{name}-Q4_0.gguf"


def batched_bench(model: Path, ngl: int, threads: int | None = None) -> list[dict]:
    cmd = [str(BIN / "llama-batched-bench"), "-m", str(model), "-c", "4608", "-b", "2048",
           "-ub", "512", "-npp", ",".join(map(str, NPP)), "-ntg", str(NTG),
           "-npl", ",".join(map(str, NPL)), "-pps", "-kvu", "-fa", "on", "-ngl", str(ngl),
           "--output-format", "jsonl"]
    if threads:
        cmd += ["-t", str(threads)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    rows = []
    for line in (out.stdout + "\n" + out.stderr).splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(out.stderr[-3000:])
    return rows


def llama_bench_tg(model: Path, ngl: int) -> dict:
    # -d 512: decode at the same 512-token depth as the batched-bench K=1 row
    cmd = [str(BIN / "llama-bench"), "-m", str(model), "-p", "0", "-n", "128", "-d", "512", "-fa", "1",
           "-ngl", str(ngl), "-r", "3", "-o", "json"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    try:
        d = json.loads(out.stdout)
        return {"avg_ts": d[0]["avg_ts"], "stddev_ts": d[0]["stddev_ts"],
                "build_commit": d[0].get("build_commit")}
    except Exception:  # noqa: BLE001
        return {"error": out.stderr[-2000:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["0.6B", "1.7B", "4B"])
    ap.add_argument("--cpu-model", default="0.6B")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--idle", type=int, default=60)
    a = ap.parse_args(argv)

    out = snapshot.new_results_dir("row_cost_llama")
    env = snapshot.full_snapshot(a.idle, [str(gguf(m)) for m in a.models])
    snapshot.write_env(out, env)
    print("results:", out, flush=True)

    arms = [(m, "metal", 99) for m in a.models]
    if a.cpu_model:
        arms.append((a.cpu_model, "cpu", 0))
    raw = []
    first = True
    for r in range(a.rounds):
        for m, backend, ngl in arms:
            if not first:
                time.sleep(a.cooldown)
            first = False
            snapshot.wait_quiet()
            before = snapshot.short_snapshot()
            rows = batched_bench(gguf(m), ngl, threads=4 if backend == "cpu" else None)
            after = snapshot.short_snapshot()
            reasons = snapshot.discard_reasons(before, after)
            for x in rows:
                raw.append({"model": m, "backend": backend, "round": r, "discard": reasons, **x})
            (out / f"round_{m}_{backend}_{r}.json").write_text(
                json.dumps({"before": before, "after": after, "discard": reasons, "rows": rows}, indent=2))
            print(f"{m} {backend} round {r} discard={reasons}", flush=True)

    tg = {}
    for m, backend, ngl in arms:
        time.sleep(a.cooldown)
        snapshot.wait_quiet()
        before = snapshot.short_snapshot()
        res = llama_bench_tg(gguf(m), ngl)
        after = snapshot.short_snapshot()
        tg[f"{m}/{backend}"] = {**res, "discard": snapshot.discard_reasons(before, after)}

    (out / "raw.json").write_text(json.dumps(raw, indent=2))
    usable = [x for x in raw if not x["discard"]] or raw
    g: dict = {}
    for x in usable:
        key = (x["model"], x["backend"], x["pp"], x["pl"])
        g.setdefault(key, []).append(x["t_tg"] / x["tg"] * 1e3)  # ms per decode step
    med = {k: statistics.median(v) for k, v in g.items()}
    table = []
    for (m, b, pp, pl), ms in sorted(med.items()):
        base = med[(m, b, pp, 1)]
        table.append({"model": m, "backend": b, "ctx": pp, "K": pl, "step_ms": round(ms, 3),
                      "c(K)": round(ms / base, 3), "serial_tok_s": round(1e3 / base, 1)})
    summary = {"c_of_K": table, "llama_bench_tg": tg,
               "all_rounds_discarded": all(x["discard"] for x in raw)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for t in table:
        print(t)
    print(json.dumps(tg, indent=2))


if __name__ == "__main__":
    main()
