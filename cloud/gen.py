"""Greedy generation of one prompt shard with vLLM (offline ``LLM.generate`` on token ids).

Temperature 0, up to 512 new tokens, thinking off (it is in the prompt template), EOS not stored.
Each shard is written atomically to tokens/shard-NNNNN.npz, so a preempted container reruns only its
shard. Sequences stuck in a repetition loop are flagged and later left out of training and eval.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

MAX_NEW = 512
EOS = (151643, 151645)  # <|endoftext|>, <|im_end|>


def is_loop(g: list[int], tail: int = 192, max_period: int = 64, frac: float = 0.9) -> bool:
    """True if the last ``tail`` tokens repeat with some period <= max_period at least ``frac`` of the time."""
    if len(g) < tail + max_period:
        return False
    a = np.asarray(g[-(tail + max_period):])
    t = a[max_period:]
    for p in range(1, max_period + 1):
        if (t == a[max_period - p:len(a) - p]).mean() >= frac:
            return True
    return False


def shard_path(root: Path, shard: int) -> Path:
    return Path(root) / "tokens" / f"shard-{shard:05d}.npz"


def run_shard(llm, root: Path, shard: int) -> dict:
    root = Path(root)
    out = shard_path(root, shard)
    stats_path = out.with_suffix(".json")
    if out.exists() and stats_path.exists():
        return json.loads(stats_path.read_text())
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    items = json.loads((root / "corpus" / "shards" / f"shard-{shard:05d}.json").read_text())
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW, detokenize=False)
    t0 = time.time()
    res = llm.generate([TokensPrompt(prompt_token_ids=it["prompt_ids"]) for it in items], sp, use_tqdm=False)
    dt = time.time() - t0
    ids, plen, glen, fin, loop, toks = [], [], [], [], [], []
    for it, r in zip(items, res):
        c = r.outputs[0]
        g = list(c.token_ids)
        stop = c.finish_reason == "stop"
        if stop and g and g[-1] in EOS:
            g = g[:-1]
        ids.append(it["id"])
        plen.append(len(it["prompt_ids"]))
        glen.append(len(g))
        fin.append(1 if stop else 0)
        loop.append((not stop) and is_loop(g))
        toks.append(np.asarray(it["prompt_ids"] + g, dtype=np.int32))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name("tmp-" + out.name)
    np.savez(tmp, id=np.array(ids, dtype=np.int64), plen=np.array(plen, dtype=np.int32),
             glen=np.array(glen, dtype=np.int32), finish_stop=np.array(fin, dtype=np.uint8),
             loop=np.array(loop, dtype=bool), tokens=np.concatenate(toks))
    os.replace(tmp, out)
    stats = {"shard": shard, "prompts": len(items), "gen_tokens": int(sum(glen)),
             "prompt_tokens": int(sum(plen)), "seconds": dt, "tok_s": sum(glen) / dt,
             "stopped_by_eos": float(np.mean(fin)), "loop_rate": float(np.mean(loop)),
             "mean_gen_tokens": float(np.mean(glen))}
    tmp = stats_path.with_name("tmp-" + stats_path.name)
    tmp.write_text(json.dumps(stats))
    os.replace(tmp, stats_path)
    return stats
