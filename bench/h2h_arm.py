"""One head-to-head arm, run as its own process: load, warm up, generate each prompt once.

Arms:
  mlx_serial     mlx-lm greedy generate_step (MLX 4-bit target)
  ours           layer-hot loop (MLX 4-bit target), config from --ours
  dflash_mlx     dflash-mlx (run with the .venv-dflash interpreter)
  llama_serial   llama-server, no draft (GGUF Q4_0 target)
  llama_dflash   llama-server, DFlash draft      (--spec-type draft-dflash  --spec-draft-n-max 15)
  llama_dflash2  llama-server, DFlash2 draft     (--spec-type draft-dflash2 --spec-draft-n-max 7)

Input: --prompts JSON list of {"id", "kind", "ids"}. Output: --out JSON with, per prompt,
the generated token ids and decode tok/s (prefill excluded).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
LLAMA_BIN = HOME / "Code" / "llama.cpp" / "build" / "bin"


# ------------------------------------------------------------------ MLX arms

def run_mlx(arm: str, model: str, prompts: list[dict], max_tokens: int, ours_cfg: dict):
    import mlx.core as mx

    from layerhot.generate import LayerHotDecoder, mlx_greedy
    from layerhot.model_utils import load

    s = load(model)
    dec = None
    if arm == "ours":
        from layerhot.collect import data_dir

        heads = data_dir(ours_cfg.get("data_model", model.split("/")[-1].replace("Qwen3-", "").replace("-4bit", ""))) / "heads"
        E, V = ours_cfg["exit"], ours_cfg.get("vocab", 16384)
        if ours_cfg["head"] == "adapter":
            from layerhot.adapter import AdapterExit

            head = AdapterExit.load(s, heads / f"adapter_E{E}_V{V}.npz").half()
        else:
            from layerhot.exit_head import AffineExit

            head = AffineExit.load(s, heads / f"affine_E{E}_V{V}.npz").half()
        dec = LayerHotDecoder(s, head, E, ours_cfg["kmax"], ours_cfg["tau"], mode=ours_cfg.get("mode", "step"))

    def gen(ids, n):
        if dec is None:
            toks, tps = mlx_greedy(s, ids, n)
            return toks, tps, {}
        toks, st = dec.generate(ids, n)
        return toks, st.tok_s, st.as_dict()

    gen(prompts[0]["ids"], 32)  # warm-up
    out = []
    for p in prompts:
        toks, tps, extra = gen(p["ids"], max_tokens)
        out.append({"id": p["id"], "tokens": toks, "tok_s": tps, **({"stats": extra} if extra else {})})
        mx.clear_cache()
    return out


def run_dflash(model: str, draft: str, prompts: list[dict], max_tokens: int):
    from dflash_mlx.engine.events import SummaryEvent, TokenEvent
    from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    from dflash_mlx.runtime.bundle import load_runtime_bundle
    from dflash_mlx.runtime.context import build_offline_runtime_context

    ctx = build_offline_runtime_context()
    b = load_runtime_bundle(model_ref=model, draft_ref=draft, draft_quant=None, verify_config=ctx.verify)
    stop = get_stop_token_ids(b.tokenizer)

    def gen(ids, n):
        stream = stream_dflash_generate(
            target_model=b.target_model, target_ops=b.target_ops, tokenizer=b.tokenizer,
            draft_model=b.draft_model, draft_backend=b.draft_backend, prompt="",
            max_new_tokens=n, use_chat_template=False, stop_token_ids=stop,
            prompt_tokens_override=list(ids), runtime_context=ctx)
        toks, summary = [], None
        try:
            for ev in stream:
                if isinstance(ev, TokenEvent):
                    toks.append(int(ev.token_id))
                elif isinstance(ev, SummaryEvent):
                    summary = ev
        finally:
            stream.close()
        el = float(summary.elapsed_us)
        pre = float(dict(summary.phase_timings_us).get("prefill", 0.0))
        tps = int(summary.generation_tokens) / ((el - pre) / 1e6) if el > pre else 0.0
        toks = [t for t in toks if t not in set(stop)]
        return toks, tps, {"acceptance": float(summary.acceptance_ratio)}

    gen(prompts[0]["ids"], 32)
    out = []
    for p in prompts:
        toks, tps, extra = gen(p["ids"], max_tokens)
        out.append({"id": p["id"], "tokens": toks, "tok_s": tps, "stats": extra})
    return out


# ------------------------------------------------------------------ llama.cpp arms

def _post(url: str, payload: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_llama(arm: str, gguf: str, draft: str | None, prompts: list[dict], max_tokens: int,
              port: int = 8089):
    cmd = [str(LLAMA_BIN / "llama-server"), "-m", gguf, "-ngl", "99", "-fa", "on", "-c", "4096",
           "-np", "1", "--port", str(port), "--no-webui"]
    if arm == "llama_dflash":
        cmd += ["-md", draft, "--spec-type", "draft-dflash", "--spec-draft-n-max", "15", "-ngld", "99"]
    elif arm == "llama_dflash2":
        cmd += ["-md", draft, "--spec-type", "draft-dflash2", "--spec-draft-n-max", "7", "-ngld", "99"]
    log = open(os.environ.get("LLAMA_LOG", "/dev/null"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:  # noqa: BLE001
                pass
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited: {proc.returncode}")
            time.sleep(1)

        def gen(ids, n):
            r = _post(base + "/completion", {"prompt": ids, "n_predict": n, "temperature": 0.0,
                                             "top_k": 1, "cache_prompt": False, "return_tokens": True,
                                             "stream": False})
            t = r.get("timings", {})
            extra = {k: t[k] for k in ("draft_n", "draft_n_accepted", "predicted_n") if k in t}
            return list(r.get("tokens", [])), float(t.get("predicted_per_second", 0.0)), extra

        gen(prompts[0]["ids"], 32)
        out = []
        for p in prompts:
            toks, tps, extra = gen(p["ids"], max_tokens)
            out.append({"id": p["id"], "tokens": toks, "tok_s": tps, "stats": extra})
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--mlx-model", default="mlx-community/Qwen3-4B-4bit")
    ap.add_argument("--gguf", default=str(HOME / "models" / "gguf" / "Qwen3-4B-Q4_0.gguf"))
    ap.add_argument("--draft", default=None, help="drafter: HF repo (dflash_mlx) or GGUF path (llama)")
    ap.add_argument("--ours", default="{}", help="JSON config for the ours arm")
    a = ap.parse_args(argv)
    prompts = json.loads(Path(a.prompts).read_text())
    t0 = time.time()
    if a.arm in ("mlx_serial", "ours"):
        res = run_mlx(a.arm, a.mlx_model, prompts, a.max_tokens, json.loads(a.ours))
    elif a.arm == "dflash_mlx":
        res = run_dflash(a.mlx_model, a.draft, prompts, a.max_tokens)
    elif a.arm.startswith("llama_"):
        res = run_llama(a.arm, a.gguf, a.draft, prompts, a.max_tokens)
    else:
        raise SystemExit(f"unknown arm {a.arm}")
    Path(a.out).write_text(json.dumps({"arm": a.arm, "wall_s": time.time() - t0, "results": res}))


if __name__ == "__main__":
    main()
