# Layer-hot decoding

A proof of concept by [Memon Systems Ltd](https://memonsystems.com). November 2026.

The work is done. On this machine the idea does not beat the existing tools, and the work stopped
at a planned gate. All code, data summaries and logs are here so the result can be checked.

## The idea

A language model reads all its weights once per generated token. While a layer's weights are
loaded, it could also run a second token through the same layer for little extra cost. The second
token is a guess, made from the first token's state after only 1-8 layers. The full model then
checks the guess. A wrong guess is thrown away, so the output is the same as plain greedy decoding.

So there are two things to measure. The first is how much a second token row costs in one pass
over the weights. The second is how often a guess from a shallow layer is right.

## Result

Qwen3 0.6B, 1.7B and 4B, 4-bit, on a MacBook Air M4 (16 GB).

| Question | Answer |
|---|---|
| Cost of a second row (MLX) | 2-11 % extra. The hardware half holds. |
| Shallow guess right (frozen model) | 27-40 % of the time, mostly punctuation and function words. Too low. |
| Layer-hot loop vs plain mlx-lm (4B, sustained load) | **0.86-0.87x**. Slower. |
| Best existing tool, tuned (llama.cpp DFlash2, Q8_0 drafter) | **1.22x** of llama.cpp serial (1.31x code, 1.13x prose) |
| Off-the-shelf EAGLE-3 on llama.cpp | 0.64-0.82x |
| Guess head with previous token's deep state (EAGLE-style) | 29 % right. Limited by training data. The layer-hot input adds nothing over the plain embedding. |

Output of the loop matches plain greedy decoding, except where the model's top two choices are tied
within 1-2 bf16 rounding steps (batched vs one-row rounding). Details in [RESULTS.md](RESULTS.md).

## Where to look

| File | What |
|---|---|
| [RESULTS.md](RESULTS.md) | Full write-up: every table, gate decision and caveat |
| [docs/PLAN.md](docs/PLAN.md) | The original plan, written before any measurement |
| [docs/PLAN_acceptance.md](docs/PLAN_acceptance.md) | Follow-up plan after step 6. Stopped at its Gate 1. |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Setup, models, the command for each step, measurement rules |
| [results/README.md](results/README.md) | Which results folder backs which number |

## Code

| Path | What |
|---|---|
| `layerhot/model_utils.py` | Runs a Qwen3 model layer by layer, with the same ops as mlx-lm |
| `layerhot/prompts.py` | Fixed prompt mix (chat, code, prose) and train/test split |
| `layerhot/collect.py` | Generates the model's own output and stores hidden states |
| `layerhot/exit_head.py` | Guess heads: logit lens and trained affine head |
| `layerhot/adapter.py` | Guess head with one attention block (step 4a) |
| `layerhot/deep_head.py` | Guess head that also reads the previous token's deep state (acceptance route) |
| `layerhot/generate.py` | The layer-hot decode loop (step 3) |
| `layerhot/pipelined.py` | Same loop with every decision on the GPU (step 6.3) |
| `bench/snapshot.py` | Machine state record and discard rules, used by every bench |
| `bench/*.py` | One script per measurement. Each has its usage in its header. |
| `tests/test_lossless.py` | Checks the loop emits the same tokens as mlx-lm greedy |
| `tools/` | Download helper, EAGLE-3 rebuild script, DFlash2 converter config |
| `results/` | Raw evidence: one folder per run, each with `env.json` |
| `env/` | Exact package versions used |

## Quick start

Needs Apple silicon and [uv](https://docs.astral.sh/uv/). About 6 minutes and 0.5 GB on an M4.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r env/requirements-main.txt

# generate Qwen3-0.6B's own output on 475 prompts, keep the layer-1 states (~5 min)
.venv/bin/python -m layerhot.collect --model 0.6B --exits 1
# train the layer-1 guess head (~15 s, ~29 % right on held-out prompts)
.venv/bin/python -m layerhot.exit_head --model 0.6B --exit 1
# run the loop on one prompt; prints tokens per sweep and compares with mlx-lm greedy
.venv/bin/python -m layerhot.generate --model 0.6B --exit 1 --kmax 2 --tau 0.3 --prompt "Explain a hash map."
```

The last line may report "identical to mlx-lm greedy: False". That happens at near-ties; the
lossless test (`tests/test_lossless.py`) checks each mismatch against the tie rule. The full runs,
llama.cpp and the drafters are in [docs/REPRODUCE.md](docs/REPRODUCE.md).

## License

MIT. Copyright (c) 2026 Memon Systems Ltd. See [LICENSE](LICENSE).
