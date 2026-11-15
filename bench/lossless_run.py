"""Run tests/test_lossless.py for one configuration and keep its output under results/.

Writes results/<ts>-lossless_<model>/{env.json, test.log, summary.json}.

Usage: python -m bench.lossless_run --model 4B --exit 1 --kmax 2 --tau 0.3 [--head affine] [--mode step]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from bench import snapshot

REPO = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="4B")
    ap.add_argument("--exit", type=int, default=1)
    ap.add_argument("--kmax", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--head", default="affine")
    ap.add_argument("--mode", default="step")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=128)
    a = ap.parse_args(argv)
    out = snapshot.new_results_dir(f"lossless_{a.model}_{a.head}_E{a.exit}_k{a.kmax}_{a.mode}")
    snapshot.write_env(out, {"short": snapshot.short_snapshot(), "software": snapshot.software(),
                             "models": [snapshot.model_info(f"mlx-community/Qwen3-{a.model}-4bit")]})
    env = {**os.environ, "LH_MODEL": a.model, "LH_EXIT": str(a.exit), "LH_KMAX": str(a.kmax),
           "LH_TAU": str(a.tau), "LH_HEAD": a.head, "LH_MODE": a.mode, "LH_N": str(a.n),
           "LH_TOKENS": str(a.tokens)}
    p = subprocess.run([str(REPO / ".venv" / "bin" / "python"), "-m", "pytest", "-q", "tests/test_lossless.py"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    log = p.stdout + p.stderr
    (out / "test.log").write_text(log)
    mism = [{"prompt": int(m[0]), "kind": m[1], "token": int(m[2]), "margin": float(m[3]), "bf16_steps": float(m[4])}
            for m in re.findall(r"mismatch prompt=(\d+) \((\w+)\) at token (\d+): margin ([\d.]+) = ([\d.]+) bf16", log)]
    summ = re.search(r"(\d+) prompts, (\d+) tokens, ([\d.]+) tokens/sweep, near-tie mismatches=(\d+), real mismatches=(\d+)", log)
    res = {"config": vars(a), "passed": p.returncode == 0, "mismatches": mism}
    if summ:
        res.update({"prompts": int(summ[1]), "tokens": int(summ[2]), "tokens_per_sweep": float(summ[3]),
                    "near_tie_mismatches": int(summ[4]), "real_mismatches": int(summ[5])})
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "mismatches"}, indent=2))
    print("results:", out)
    return 0 if p.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
