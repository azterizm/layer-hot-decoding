"""Step 6.1: does a quantized DFlash drafter turn accepted tokens into speed on this M4?

Each variant (backend + drafter precision) runs in its own process and is compared A-B-B-A with
plain serial decoding of the same backend, on the head-to-head prompts, after a warm-up that brings
the GPU to a steady state. Ratio per prompt = mean(spec) / mean(serial).

  llama.cpp: two llama-servers side by side (target alone; target + drafter), requests alternate.
             (Per-request speculative settings are compiled out of llama-server.)
  dflash-mlx: one process holds a stock mlx-lm target (serial) and a dflash-mlx bundle with the
              drafter quantized by dflash-mlx itself (`draft_quant`: none, w8, w4).

Usage: python -m bench.draft_quant [--variants ...] [--max-tokens 256] [--warmup-s 60]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from bench import snapshot

REPO = Path(__file__).resolve().parents[1]
GGUF = Path.home() / "models" / "gguf"
LLAMA_BIN = Path.home() / "Code" / "llama.cpp" / "build" / "bin"
H2H_PROMPTS = REPO / "results" / "20261114-234015-head_to_head" / "prompts.json"

VARIANTS = {
    # name: (backend, drafter, spec type, n_max)
    "llama_dflash_bf16": ("llama", "Qwen3-4B-DFlash-b16.gguf", "draft-dflash", 15),
    "llama_dflash_q8": ("llama", "Qwen3-4B-DFlash-b16-Q8_0.gguf", "draft-dflash", 15),
    "llama_dflash_q4": ("llama", "Qwen3-4B-DFlash-b16-Q4_0.gguf", "draft-dflash", 15),
    "llama_dflash2_bf16": ("llama", "Qwen3-4B-DFlash2.gguf", "draft-dflash2", 7),
    "llama_dflash2_q8": ("llama", "Qwen3-4B-DFlash2-Q8_0.gguf", "draft-dflash2", 7),
    "llama_dflash2_q4": ("llama", "Qwen3-4B-DFlash2-Q4_0.gguf", "draft-dflash2", 7),
    # verify width: fewer draft tokens per block = fewer rows per verify
    "llama_dflash_q8_n7": ("llama", "Qwen3-4B-DFlash-b16-Q8_0.gguf", "draft-dflash", 7),
    "llama_dflash_q8_n3": ("llama", "Qwen3-4B-DFlash-b16-Q8_0.gguf", "draft-dflash", 3),
    "mlx_dflash_bf16": ("mlx", "none", None, None),
    "mlx_dflash_w8": ("mlx", "w8", None, None),
    "mlx_dflash_w4": ("mlx", "w4", None, None),
}


# ------------------------------------------------------------------ llama.cpp

def _post(url, payload, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _server(port, draft=None, spec=None, n_max=None, log=None):
    cmd = [str(LLAMA_BIN / "llama-server"), "-m", str(GGUF / "Qwen3-4B-Q4_0.gguf"), "-ngl", "99", "-fa", "on",
           "-c", "4096", "-np", "1", "--port", str(port), "--no-webui"]
    if draft:
        cmd += ["-md", str(GGUF / draft), "-ngld", "99", "--spec-type", spec, "--spec-draft-n-max", str(n_max)]
    p = subprocess.Popen(cmd, stdout=open(log or os.devnull, "w"), stderr=subprocess.STDOUT)
    for _ in range(300):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return p
        except Exception:  # noqa: BLE001
            pass
        if p.poll() is not None:
            raise RuntimeError(f"llama-server exited ({cmd})")
        time.sleep(1)
    raise RuntimeError("llama-server did not start")


def run_llama(name, prompts, max_tokens, warmup_s, out):
    _, draft, spec, n_max = VARIANTS[name]
    ps = _server(8091, log=str(out / f"{name}_serial.log"))
    pd = _server(8092, draft, spec, n_max, log=str(out / f"{name}_spec.log"))

    def gen(port, ids, n):
        r = _post(f"http://127.0.0.1:{port}/completion",
                  {"prompt": ids, "n_predict": n, "temperature": 0.0, "top_k": 1, "cache_prompt": False,
                   "return_tokens": True})
        t = r["timings"]
        return list(r["tokens"]), float(t["predicted_per_second"]), {k: t.get(k) for k in ("draft_n", "draft_n_accepted", "predicted_n")}

    try:
        t0 = time.time()
        while time.time() - t0 < warmup_s:
            gen(8091, prompts[0]["ids"], 64)
            gen(8092, prompts[0]["ids"], 64)
        rows = []
        for p in prompts:
            a1, sa1, _ = gen(8091, p["ids"], max_tokens)
            b1, sb1, st1 = gen(8092, p["ids"], max_tokens)
            b2, sb2, _ = gen(8092, p["ids"], max_tokens)
            a2, sa2, _ = gen(8091, p["ids"], max_tokens)
            n, acc = st1.get("predicted_n"), st1.get("draft_n_accepted")
            rows.append({"id": p["id"], "kind": p["kind"], "serial_tok_s": [sa1, sa2], "spec_tok_s": [sb1, sb2],
                         "ratio": (sb1 + sb2) / (sa1 + sa2), "identical": b1 == a1,
                         "tokens_per_target_pass": n / (n - acc) if n and acc is not None and n > acc else None})
            print(name, rows[-1], flush=True)
        return rows
    finally:
        for p in (ps, pd):
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()


# ------------------------------------------------------------------ dflash-mlx (runs under .venv-dflash)

def run_mlx(name, prompts, max_tokens, warmup_s, out):
    import mlx.core as mx
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context
    from mlx_lm.generate import generate_step
    from mlx_lm.utils import load as load_pristine_target  # what dflash-mlx's own benchmark uses

    quant = VARIANTS[name][1]
    # stock target for the serial arm, loaded the way dflash-mlx's own benchmark loads its baseline
    model, tok = load_pristine_target("mlx-community/Qwen3-4B-4bit")
    ctx = build_offline_runtime_context()
    b = load_runtime_bundle(model_ref="mlx-community/Qwen3-4B-4bit", draft_ref="z-lab/Qwen3-4B-DFlash-b16",
                            draft_quant=quant, verify_config=ctx.verify)
    stop = set(get_stop_token_ids(b.tokenizer))
    eos = set(tok.eos_token_ids)

    def serial(ids, n):
        out_, t_first = [], None
        for i, (t, _) in enumerate(generate_step(mx.array(ids), model, max_tokens=n)):
            if i == 0:
                t_first = time.perf_counter()
            t = int(t)
            if t in eos:
                break
            out_.append(t)
        return out_, (len(out_) - 1) / (time.perf_counter() - t_first)

    def spec(ids, n):
        stream = stream_dflash_generate(
            target_model=b.target_model, target_ops=b.target_ops, tokenizer=b.tokenizer,
            draft_model=b.draft_model, draft_backend=b.draft_backend, prompt="", max_new_tokens=n,
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
        el, pre = float(summ.elapsed_us), float(dict(summ.phase_timings_us).get("prefill", 0.0))
        return [t for t in toks if t not in stop], int(summ.generation_tokens) / ((el - pre) / 1e6), float(summ.acceptance_ratio)

    t0 = time.time()
    while time.time() - t0 < warmup_s:
        serial(prompts[0]["ids"], 64)
        spec(prompts[0]["ids"], 64)
    rows = []
    for p in prompts:
        a1, sa1 = serial(p["ids"], max_tokens)
        b1, sb1, acc = spec(p["ids"], max_tokens)
        _, sb2, _ = spec(p["ids"], max_tokens)
        _, sa2 = serial(p["ids"], max_tokens)
        rows.append({"id": p["id"], "kind": p["kind"], "serial_tok_s": [sa1, sa2], "spec_tok_s": [sb1, sb2],
                     "ratio": (sb1 + sb2) / (sa1 + sa2), "identical": b1 == a1,
                     "tokens_per_target_pass": 1 / (1 - acc) if acc < 1 else None})
        print(name, rows[-1], flush=True)
        mx.clear_cache()
    return rows


# ------------------------------------------------------------------ orchestration

def summarize(rows):
    res = {}
    for k in ("code", "prose", "all"):
        sel = [r for r in rows if k == "all" or r["kind"] == k]
        tpp = [r["tokens_per_target_pass"] for r in sel if r["tokens_per_target_pass"]]
        res[k] = {"ratio": statistics.mean(r["ratio"] for r in sel),
                  "spec_tok_s": statistics.mean(statistics.mean(r["spec_tok_s"]) for r in sel),
                  "serial_tok_s": statistics.mean(statistics.mean(r["serial_tok_s"]) for r in sel),
                  "tokens_per_target_pass": statistics.mean(tpp) if tpp else None,
                  "identical": sum(r["identical"] for r in sel), "n": len(sel)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--warmup-s", type=int, default=60)
    ap.add_argument("--cooldown", type=int, default=60)
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)  # internal: run one variant
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    prompts = json.loads(H2H_PROMPTS.read_text())

    if a.one:  # child process
        out = Path(a.out)
        fn = run_llama if VARIANTS[a.one][0] == "llama" else run_mlx
        rows = fn(a.one, prompts, a.max_tokens, a.warmup_s, out)
        (out / f"{a.one}.json").write_text(json.dumps(rows, indent=1))
        return 0

    out = snapshot.new_results_dir("draft_quant")
    snapshot.write_env(out, snapshot.full_snapshot(60, [str(GGUF / "Qwen3-4B-Q4_0.gguf"), "mlx-community/Qwen3-4B-4bit",
                                                       "z-lab/Qwen3-4B-DFlash-b16"]
                                                   + [str(GGUF / VARIANTS[v][1]) for v in a.variants if VARIANTS[v][0] == "llama"]))
    print("results:", out, flush=True)
    summary = {}
    for i, v in enumerate(a.variants):
        if i:
            time.sleep(a.cooldown)
        snapshot.wait_quiet()
        before = snapshot.short_snapshot()
        py = REPO / (".venv-dflash" if VARIANTS[v][0] == "mlx" else ".venv") / "bin" / "python"
        p = subprocess.run([str(py), "-u", "-m", "bench.draft_quant", "--one", v, "--out", str(out),
                            "--max-tokens", str(a.max_tokens), "--warmup-s", str(a.warmup_s)],
                           cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True)
        after = snapshot.short_snapshot()
        reasons = snapshot.discard_reasons(before, after)
        if p.returncode != 0:
            print(f"{v} FAILED:\n{p.stderr[-3000:]}", flush=True)
            continue
        rows = json.loads((out / f"{v}.json").read_text())
        summary[v] = {**summarize(rows), "discard": reasons}
        (out / f"{v}_env.json").write_text(json.dumps({"before": before, "after": after, "discard": reasons}, indent=2))
        s = summary[v]["all"]
        print(f"{v:20s} ratio {s['ratio']:.3f} (code {summary[v]['code']['ratio']:.3f}, prose {summary[v]['prose']['ratio']:.3f})"
              f"  spec {s['spec_tok_s']:.1f} vs serial {s['serial_tok_s']:.1f} tok/s  tokens/pass {s['tokens_per_target_pass'] or 0:.2f}"
              f"  identical {s['identical']}/{s['n']}  discard={reasons}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
