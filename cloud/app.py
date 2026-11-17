"""Phase 0 on Modal: cloud retrial of the layer-hot head (docs/PLAN_sram_expansion.md, section 0).

Run from the repo root with the venv's modal (Python 3.12):

    modal run -m cloud.app::step1_selftest      # CPU tests in a container
    modal run -m cloud.app::step2_prepare       # model + datasets to the volume, corpus, anchor upload
    modal run -m cloud.app::step3_pilot         # anchor check + vLLM on the val/test shards (2 H100)
    modal run --detach -m cloud.app::step4_generate --n-train-shards N
    modal run --detach -m cloud.app::step5_train
    modal run -m cloud.app::step6_evaluate      # fixes the gate rule from validation, then scores the test set once
    modal run -m cloud.app::step7_fetch         # small JSON results to results/<timestamp>-phase0/
    modal run -m cloud.app::status              # spend by stage

Every stage checks the ledger first: no stage starts if spend so far plus its estimate passes the cap.
"""

import hashlib
import json
import math
import sys
from pathlib import Path

import modal

ROOT = "/vol"
MODEL = "Qwen/Qwen3-4B"
MODEL_REV = "1cfa9a7208912126459214e8b04321603b3df60c"
ENV = {"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
VAL_TEST_SHARD = 1000
TRAIN_SHARD = 2000

app = modal.App("layerhot-phase0")
vol = modal.Volume.from_name("layerhot-phase0", create_if_missing=True, version=2)

torch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.8.0", "transformers==4.57.1", "numpy==2.3.3", "pyarrow==21.0.0",
                    "huggingface_hub==0.35.3", "pytest==8.4.2")
    .env(ENV)
    .add_local_dir("tests", "/root/tests")
    .add_local_python_source("cloud", "layerhot")
)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.11.0", "transformers==4.57.1", "huggingface_hub==0.35.3")
    .env(ENV)
    .add_local_python_source("cloud", "layerhot")
)
VOLS = {ROOT: vol}


def _meter(stage, gpu, cpu, mem_gib, n_parallel=1):
    from cloud.ledger import Meter

    return Meter(Path(ROOT), stage, gpu, cpu, mem_gib, n_parallel=n_parallel)


# ------------------------------------------------------------------ CPU stages

@app.function(image=torch_image, volumes=VOLS, timeout=1800, cpu=2, memory=8192)
def selftest() -> tuple[int, str]:
    import subprocess

    m = _meter("selftest", None, 2, 8)
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "tests/test_cloud.py", "tests/test_cloud_corpus.py"],
                           capture_output=True, text=True, cwd="/root")
        return r.returncode, (r.stdout[-8000:] + "\n" + r.stderr[-4000:])
    finally:
        m.close()


@app.function(image=torch_image, volumes=VOLS, timeout=3600, cpu=4, memory=16384)
def prepare(local_hash: str, n_total: int) -> dict:
    """Download the pinned model and datasets, check the anchor prompts, build the corpus and shards."""
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from cloud import corpus
    from cloud.train import write_json
    from layerhot import prompts as P

    m = _meter("prepare", None, 4, 16)
    try:
        root = Path(ROOT)
        snapshot_download(MODEL, revision=MODEL_REV, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        corpus.fetch()
        items = P.build()
        h = hashlib.sha256(json.dumps([[it["id"], it["kind"], it["text"], it["split"], it["group"]]
                                       for it in items]).encode()).hexdigest()
        tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
        built = corpus.build(tok, seed=0, n_total=n_total)
        d = root / "corpus"
        (d / "shards").mkdir(parents=True, exist_ok=True)
        with open(d / "meta.jsonl", "w") as f:
            for it in built:
                f.write(json.dumps({k: it[k] for k in ("id", "kind", "source", "group", "split")}
                                   | {"lang": it.get("lang", "")}) + "\n")
        with open(d / "items.jsonl", "w") as f:
            for it in built:
                f.write(json.dumps(it) + "\n")
        vt = [it for it in built if it["split"] == "val"] + [it for it in built if it["split"] == "test"]
        tr = [it for it in built if it["split"] == "train"]
        shards = [vt[i:i + VAL_TEST_SHARD] for i in range(0, len(vt), VAL_TEST_SHARD)]
        shards += [tr[i:i + TRAIN_SHARD] for i in range(0, len(tr), TRAIN_SHARD)]
        for k, sh in enumerate(shards):
            (d / "shards" / f"shard-{k:05d}.json").write_text(
                json.dumps([{"id": it["id"], "prompt_ids": it["prompt_ids"]} for it in sh]))
        info = {"summary": corpus.summary(built), "notes": list(corpus.NOTES), "n_shards": len(shards),
                "val_test_shards": math.ceil(len(vt) / VAL_TEST_SHARD), "shard_sizes": [len(s) for s in shards],
                "anchor_prompts_hash": h, "anchor_prompts_hash_local": local_hash,
                "anchor_prompts_match": h == local_hash}
        write_json(d / "summary.json", info)
        return info
    finally:
        m.close()


# ------------------------------------------------------------------ generation (vLLM)

@app.cls(image=vllm_image, gpu="H100", volumes=VOLS, timeout=3600, cpu=4, memory=16384,
         max_containers=8, retries=modal.Retries(max_retries=1, initial_delay=5.0))
class Generator:
    n_parallel: int = modal.parameter(default=1)

    @modal.enter()
    def load(self):
        from vllm import LLM

        self.meter = _meter("generate", "H100", 4, 16, n_parallel=self.n_parallel)
        self.llm = LLM(model=MODEL, revision=MODEL_REV, tokenizer_revision=MODEL_REV, dtype="bfloat16",
                       max_model_len=2048, gpu_memory_utilization=0.90, max_num_seqs=512, seed=0,
                       enable_prefix_caching=False)

    @modal.method()
    def run(self, shard: int) -> dict:
        from cloud import gen

        if not self.meter.guard():
            return {"shard": shard, "skipped": "cost cap"}
        r = gen.run_shard(self.llm, Path(ROOT), shard)
        vol.commit()
        return r

    @modal.exit()
    def close(self):
        self.meter.close()


@app.function(image=torch_image, volumes=VOLS, timeout=3 * 3600, cpu=1, memory=2048)
def gen_all(shards: list[int], n_parallel: int) -> list[dict]:
    g = Generator(n_parallel=n_parallel)
    out = []
    it = (g.run.remote(s) for s in shards) if n_parallel == 1 else \
        g.run.map(shards, return_exceptions=True, order_outputs=False)
    for r in it:
        out.append(r if isinstance(r, dict) else {"error": repr(r)})
        print(json.dumps(out[-1]), flush=True)
    return out


# ------------------------------------------------------------------ GPU stages (torch)

@app.function(image=torch_image, gpu="H100", volumes=VOLS, timeout=3600, cpu=4, memory=32768)
def pilot_anchor(local_hash: str) -> dict:
    from cloud import train

    m = _meter("pilot", "H100", 4, 32)
    try:
        return train.pilot(Path(ROOT), local_hash)
    finally:
        m.close()


@app.function(image=torch_image, gpu="H100", volumes=VOLS, timeout=4 * 3600, cpu=4, memory=32768)
def train_main(run_name: str = "main", max_seqs: int = 0) -> dict:
    from cloud import train

    m = _meter("train", "H100", 4, 32)
    try:
        return train.run(Path(ROOT), run_name, meter=m, max_seqs=max_seqs, commit=vol.commit)
    finally:
        m.close()


@app.function(image=torch_image, gpu="H100", volumes=VOLS, timeout=3600, cpu=4, memory=32768)
def evaluate_final(run_name: str = "main", allow_stopped: bool = False) -> dict:
    from cloud import evaluate

    m = _meter("evaluate", "H100", 4, 32)
    try:
        return evaluate.final(Path(ROOT), run_name, allow_stopped=allow_stopped)
    finally:
        m.close()


# ------------------------------------------------------------------ local side

def _read(path: str) -> bytes:
    return b"".join(vol.read_file(path))


def _spent() -> tuple[float, dict]:
    from cloud import ledger

    recs = []
    try:
        for e in vol.listdir("/ledger"):
            if e.path.endswith(".json"):
                recs.append(json.loads(_read(e.path)))
    except Exception:  # no ledger yet
        pass
    return ledger.total(recs), ledger.by_stage(recs)


def _check(stage: str, estimate: float) -> float:
    from cloud import ledger

    spent, _ = _spent()
    ledger.check(spent, estimate, stage)
    print(f"[ledger] spent ${spent:.2f}, {stage} estimate ${estimate:.2f}, cap ${ledger.CAP_USD:.2f}")
    return spent


def _local_hash() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from layerhot import prompts as P

    items = P.build()
    return hashlib.sha256(json.dumps([[it["id"], it["kind"], it["text"], it["split"], it["group"]]
                                      for it in items]).encode()).hexdigest()


def _h100_hour(cpu: float, mem: float) -> float:
    from cloud import ledger

    return ledger.rate_per_s("H100", cpu, mem) * 3600


@app.local_entrypoint()
def status():
    total, by = _spent()
    for k, v in sorted(by.items()):
        print(f"{k:10s} ${v['usd']:7.2f}  GPU-h {v['gpu_hours']:.2f}  containers {v['containers']}")
    print(f"total      ${total:7.2f}")


@app.local_entrypoint()
def step1_selftest():
    _check("selftest", 0.1)
    code, out = selftest.remote()
    print(out)
    if code:
        raise SystemExit(f"selftest failed ({code})")


@app.local_entrypoint()
def step2_prepare(n_total: int = 130_000):
    _check("prepare", 0.5)
    with vol.batch_upload(force=True) as b:
        b.put_file("data/4B/gen.jsonl", "/anchor/gen.jsonl")
    info = prepare.remote(_local_hash(), n_total)
    print(json.dumps(info, indent=2))


@app.local_entrypoint()
def step3_pilot():
    _check("pilot", 2 * _h100_hour(4, 32) * 0.5)
    info = json.loads(_read("/corpus/summary.json"))
    vt = list(range(info["val_test_shards"]))
    h = _local_hash()
    a = pilot_anchor.spawn(h)
    g = gen_all.spawn(vt, 1)
    gen = g.get()
    print(json.dumps(gen, indent=2))
    res = a.get()
    print(json.dumps({k: v for k, v in res.items() if k != "dry_run"}, indent=2))
    print(json.dumps(res["dry_run"], indent=2))


@app.local_entrypoint()
def step4_generate(n_train_shards: int = 0, n_parallel: int = 8, estimate: float = 0.0):
    info = json.loads(_read("/corpus/summary.json"))
    first = info["val_test_shards"]
    last = info["n_shards"] if not n_train_shards else min(info["n_shards"], first + n_train_shards)
    shards = list(range(first, last))
    _check("generate", estimate or 20.0)
    res = gen_all.remote(shards, n_parallel)
    ok = [r for r in res if "gen_tokens" in r]
    print(f"{len(ok)}/{len(shards)} shards, {sum(r['gen_tokens'] for r in ok):,} generated tokens")
    bad = [r for r in res if "gen_tokens" not in r]
    if bad:
        print("not done:", json.dumps(bad, indent=2))


@app.local_entrypoint()
def step5_train(run_name: str = "main", estimate: float = 0.0, max_seqs: int = 0):
    _check("train", estimate or 8.0)
    print(json.dumps(train_main.remote(run_name, max_seqs), indent=2))


@app.local_entrypoint()
def step6_evaluate(run_name: str = "main", out_dir: str = ""):
    _check("evaluate", 1.0)
    res = evaluate_final.remote(run_name)
    print(json.dumps({k: res[k] for k in ("gate_rule", "gate", "bootstrap", "seed_spread_A_parity_lr1e-3")},
                     indent=2))


@app.local_entrypoint()
def step7_fetch(run_name: str = "main", tag: str = "phase0", include_test: bool = True):
    """Copy the small JSON results from the volume into results/<timestamp>-<tag>/."""
    import time as _t

    from cloud import corpus, ledger

    out = Path("results") / (_t.strftime("%Y%m%d-%H%M%S") + f"-{tag}")
    out.mkdir(parents=True)
    want = ["/corpus/summary.json", "/pilot/anchor.json", f"/train/{run_name}/plan.json",
            f"/train/{run_name}/summary.json", f"/train/{run_name}/curve.jsonl"]
    if include_test:
        want += [f"/results/{run_name}/gate_rule.json", f"/results/{run_name}/evaluate.json"]
    for p in want:
        try:
            (out / p.strip("/").replace("/", "_").replace(f"_{run_name}_", "_")).write_bytes(_read(p))
        except Exception as e:  # not produced yet
            print("missing", p, type(e).__name__)
    shards = []
    try:
        for e in vol.listdir("/tokens"):
            if e.path.endswith(".json") and "/tmp-" not in e.path:
                shards.append(json.loads(_read(e.path)))
    except Exception:
        pass
    shards.sort(key=lambda r: r["shard"])
    gen = {"shards": shards, "prompts": sum(r["prompts"] for r in shards),
           "gen_tokens": sum(r["gen_tokens"] for r in shards),
           "loop_rate": (sum(r["loop_rate"] * r["prompts"] for r in shards) / max(1, sum(r["prompts"] for r in shards))),
           "stopped_by_eos": (sum(r["stopped_by_eos"] * r["prompts"] for r in shards)
                              / max(1, sum(r["prompts"] for r in shards)))}
    (out / "generation.json").write_text(json.dumps(gen, indent=2))
    total, by = _spent()
    env = {"where": "Modal", "app": app.name, "volume": "layerhot-phase0", "gpu": "H100 80GB",
           "modal_client": modal.__version__, "model": {"repo": MODEL, "revision": MODEL_REV, "dtype": "bfloat16"},
           "datasets": corpus.REVISIONS,
           "images": {"torch": ["torch==2.8.0", "transformers==4.57.1", "numpy==2.3.3", "pyarrow==21.0.0"],
                      "vllm": ["vllm==0.11.0", "transformers==4.57.1", "cuda 12.8.1"]},
           "spend_usd": {"total": total, "cap": ledger.CAP_USD, "by_stage": by}}
    (out / "env.json").write_text(json.dumps(env, indent=2))
    print("results:", out)
