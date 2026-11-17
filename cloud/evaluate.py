"""Final scoring of the Phase 0 heads and the Phase 0 gate.

The gate rule is written to the volume from validation numbers alone, before the test set is scored:
- the best head by validation top-1 (h_1 and e arms only) must reach 65.0 % top-1 overall or 75.0 % on code;
- on that head's architecture, at the learning rate with the best mean validation top-1 of the two arms
  (seed 0), h_1 must beat e by 3.0 points or more on the test set. Under 1.0 point means the layer-hot
  input is disproven; 1.0 to 3.0 fails the gate as a small effect.
Both must pass to revive layer-hot.
"""

from __future__ import annotations

import json
import string
import time
from pathlib import Path

import numpy as np
import torch

from cloud import heads as Hd
from cloud import train as Tr
from cloud.teacher import Teacher

CYCLE_SUSTAINED = 1.36  # one-guess cycle cost in one-row steps, chained loop under sustained load
CYCLE_GPU = 1.09  # same cycle, GPU work only (bench/cycle_parts.py)
TEST_PER_SEQ = 50
GATE = {"model_top1": 0.65, "model_code_top1": 0.75, "hyp_pass_points": 3.0, "hyp_disproven_points": 1.0}
THRESHOLDS = (0.0, 0.3, 0.5, 0.7, 0.9)

# copied from bench/acceptance.py (that file imports MLX at the top)
FUNCTION_WORDS = set("""a an the of to in on at by for with from as and or but if then than that this these
those is are was were be been being it its it's he she they we you i me my your our their his her him them
not no so do does did have has had will would can could should may might must there here which who whom
what when where why how all any each some such into over under about after before between through during
up down out off again also just only very more most other same own too s t""".split())


def token_classes(tok, ids_next: np.ndarray, ids_prev: np.ndarray) -> np.ndarray:
    cache: dict[int, str] = {}

    def dec(i: int) -> str:
        if i not in cache:
            cache[i] = tok.decode([int(i)])
        return cache[i]

    punct = set(string.punctuation) | {"’", "“", "”", "—", "–", "…"}
    out = []
    for n, p in zip(ids_next, ids_prev):
        s = dec(n)
        st = s.strip()
        if not st or all((c in punct) or not c.isalnum() for c in st):
            out.append("punct/ws")
        elif st.lower() in FUNCTION_WORDS and (s[0].isspace() or not dec(p)[-1:].isalnum()):
            out.append("function")
        elif s[0].isalnum() and dec(p)[-1:].isalnum():
            out.append("subword")
        else:
            out.append("content")
    return np.array(out)


def choose(val: dict, cfgs: list[dict]) -> dict:
    """The gate's choices, from validation top-1 only."""
    cand = [c for c in cfgs if c["arm"] in ("h1", "e")]
    best = max(cand, key=lambda c: (val[c["name"]], c["name"]))
    arch = best["arch"]
    by = {(c["arm"], c["lr"]): c["name"] for c in cand if c["arch"] == arch and c["seed"] == 0}
    lrs = sorted({lr for (_, lr) in by})
    lr = max(lrs, key=lambda lr: (val[by[("h1", lr)]] + val[by[("e", lr)]]) / 2)
    return {"best": best["name"], "best_val_top1": val[best["name"]], "arch": arch, "lr": lr,
            "h1": by[("h1", lr)], "e": by[("e", lr)], "thresholds": GATE, "rule": __doc__}


def decide(choice: dict, test: dict) -> dict:
    b = test[choice["best"]]
    model_pass = b["top1"] >= GATE["model_top1"] or b["by_kind"].get("code", 0.0) >= GATE["model_code_top1"]
    d = 100 * (test[choice["h1"]]["top1"] - test[choice["e"]]["top1"])
    if d >= GATE["hyp_pass_points"]:
        hyp = "pass"
    elif d < GATE["hyp_disproven_points"]:
        hyp = "disproven"
    else:
        hyp = "small effect (fails)"
    return {"model_gate": "pass" if model_pass else "fail", "best_top1": b["top1"],
            "best_code_top1": b["by_kind"].get("code"), "h1_minus_e_points": d, "hypothesis_gate": hyp,
            "revive_layer_hot": bool(model_pass and hyp == "pass")}


def bootstrap_diff(hit_a: np.ndarray, hit_b: np.ndarray, groups: np.ndarray, n: int = 2000,
                   seed: int = 0) -> dict:
    """Paired bootstrap over groups of top-1(a) - top-1(b), in points: mean and 95 % interval."""
    g_ids, g = np.unique(groups, return_inverse=True)
    G = len(g_ids)
    ha = np.bincount(g, weights=hit_a, minlength=G)
    hb = np.bincount(g, weights=hit_b, minlength=G)
    cnt = np.bincount(g, minlength=G).astype(np.float64)
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(n)]).astype(np.float64)
    d = 100 * (W @ ha - W @ hb) / (W @ cnt)
    point = 100 * (ha.sum() - hb.sum()) / cnt.sum()
    return {"points": float(point), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "groups": int(G), "resamples": n}


def metrics(tok: np.ndarray, conf: np.ndarray, y: np.ndarray, kind: np.ndarray, cls: np.ndarray) -> dict:
    hit = tok == y
    acc = float(hit.mean())
    pr = []
    for t in THRESHOLDS:
        m = (tok >= 0) & (conf >= t)
        pr.append({"tau": t, "coverage": float(m.mean()), "precision": float(hit[m].mean()) if m.any() else None})
    return {"top1": acc, "n": int(len(y)), "guess_coverage": float((tok >= 0).mean()),
            "by_kind": {k: float(hit[kind == k].mean()) for k in np.unique(kind)},
            "by_class": {c: float(hit[cls == c].mean()) for c in np.unique(cls)},
            "pr_curve": pr,
            "pred_speed_one_guess": {"tokens_per_cycle": 1 + acc,
                                     "at_sustained_cycle_cost": (1 + acc) / CYCLE_SUSTAINED,
                                     "at_gpu_only_cycle_cost": (1 + acc) / CYCLE_GPU}}


@torch.no_grad()
def score(teacher, heads, cfgs, seqs, plens, picks, device="cuda") -> dict:
    """Predictions of every head on the picked rows, plus per-row label, current token and sequence index."""
    pred = {n: ([], []) for n in heads}
    y, prev, seq, pos = [], [], [], []
    for ch in Tr.eval_chunks(teacher, seqs, plens, picks):
        p = Tr.predict_chunk(teacher, heads, cfgs, ch, device, amp=True)
        s, t = ch["sel"].nonzero(as_tuple=True)
        y.append(ch["label"][s, t].cpu())
        prev.append(ch["ids"][s, t].cpu())
        seq.append(torch.as_tensor(ch["js"])[s.cpu()])
        pos.append(t.cpu())
        for n, (tk, cf) in p.items():
            pred[n][0].append(tk.cpu())
            pred[n][1].append(cf.float().cpu())
    out = {"y": torch.cat(y).numpy(), "prev": torch.cat(prev).numpy(), "seq": torch.cat(seq).numpy(),
           "pos": torch.cat(pos).numpy()}
    out["pred"] = {n: (torch.cat(a).numpy(), torch.cat(b).numpy()) for n, (a, b) in pred.items()}
    return out


def load_heads(rdir: Path, teacher: Teacher, out: Hd.OutSlice, device="cuda"):
    heads, cfgs = {}, {}
    for f in sorted((rdir / "final").glob("*.pt")):
        d = torch.load(f, map_location=device, weights_only=False)
        c = d["cfg"]
        h = Hd.make(c["arch"], c["arm"], out, teacher.H, teacher.eps, c["seed"]).to(device)
        h.load_state_dict(d["state"])
        h.eval()
        heads[c["name"]] = h
        cfgs[c["name"]] = c
    return heads, cfgs


def final(root: Path, run_name: str = "main", allow_stopped: bool = False, log=print) -> dict:
    import hashlib

    root = Path(root)
    torch.backends.cuda.matmul.allow_tf32 = True
    rdir = root / "train" / run_name
    odir = root / "results" / run_name
    raw = (rdir / "summary.json").read_bytes()
    tsum = json.loads(raw)
    if tsum.get("stopped") and not allow_stopped:
        raise RuntimeError(f"training stopped early ({tsum['stopped']}); pass allow_stopped to score anyway")
    plan = json.loads((rdir / "plan.json").read_text())
    choice = choose(tsum["val_top1"], plan["configs"])
    choice["train_summary_sha256"] = hashlib.sha256(raw).hexdigest()
    choice["written_at"] = time.time()
    rule_path = odir / "gate_rule.json"
    if not rule_path.exists():  # the rule is fixed once, before any test score exists
        Tr.write_json(rule_path, choice)
    choice = json.loads(rule_path.read_text())
    if choice["train_summary_sha256"] != hashlib.sha256(raw).hexdigest():
        raise RuntimeError("gate_rule.json belongs to another training run; use a new run name")
    log(f"gate rule fixed: best {choice['best']}, pair {choice['h1']} vs {choice['e']}")

    from transformers import AutoTokenizer
    from cloud.teacher import MODEL, REVISION

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    teacher = Teacher(device="cuda")
    rows = np.load(rdir / "vocab_rows.npy")
    out = Hd.OutSlice(teacher.norm_w, teacher.eps, teacher.lm_w, rows, teacher.vocab_size)
    heads, cfgs = load_heads(rdir, teacher, out)

    tok = Tr.load_tokens(root)
    meta = Tr.load_meta(root)
    te = Tr.usable(tok, meta, "test")
    picks = [Tr.pick_positions(int(tok["plen"][k]), int(tok["glen"][k]), TEST_PER_SEQ, 10**7 + int(tok["id"][k]))
             for k in te]
    t0 = time.time()
    S = score(teacher, heads, cfgs, [tok["seq"][k] for k in te], [int(tok["plen"][k]) for k in te], picks)
    log(f"test scored: {len(S['y'])} rows from {len(te)} sequences ({time.time() - t0:.0f}s)")
    ids = tok["id"][te][S["seq"]]
    kind = np.array([meta[int(i)]["kind"] for i in ids])
    group = np.array([meta[int(i)]["group"] for i in ids])
    cls = token_classes(tokenizer, S["y"], S["prev"])
    test = {n: metrics(*S["pred"][n], S["y"], kind, cls) for n in heads}
    in_slice = float(np.isin(S["y"], rows).mean())

    # anchor: the Mac run's test split, every row, labels from this bf16 model
    gen = Tr.anchor_data(root)
    at = [g for g in gen if g["split"] == "test"]
    a_seqs = [np.array(g["prompt_ids"] + g["gen_ids"], dtype=np.int64) for g in at]
    a_pl = [len(g["prompt_ids"]) for g in at]
    A = score(teacher, heads, cfgs, a_seqs, a_pl, [np.arange(p, len(s) - 1) for s, p in zip(a_seqs, a_pl)])
    a_kind = np.array([at[j]["kind"] for j in A["seq"]])
    a_cls = token_classes(tokenizer, A["y"], A["prev"])
    anchor = {n: metrics(*A["pred"][n], A["y"], a_kind, a_cls) for n in heads}

    gate = decide(choice, test)
    hit = {n: (S["pred"][n][0] == S["y"]).astype(np.float64) for n in heads}
    boot = {"gate_pair": bootstrap_diff(hit[choice["h1"]], hit[choice["e"]], group)}
    for c in plan["configs"]:
        if c["arm"] == "h1" and c["name"].replace("/h1/", "/e/") in hit:
            boot[c["name"].replace("/h1/", "/h1-e/")] = bootstrap_diff(hit[c["name"]],
                                                                      hit[c["name"].replace("/h1/", "/e/")], group)
    seeds = {}
    for arm in ("h1", "e"):
        names = [c["name"] for c in plan["configs"]
                 if c["arch"] == "A-parity" and c["arm"] == arm and c["lr"] == 1e-3]
        v = [test[n]["top1"] for n in names]
        seeds[arm] = {"names": names, "top1": v, "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                      "range": float(max(v) - min(v)) if v else None}
    pairs = [100 * (a - b) for a, b in zip(seeds["h1"]["top1"], seeds["e"]["top1"])]
    seeds["h1_minus_e_points_by_seed"] = pairs

    res = {"gate_rule": choice, "gate": gate, "bootstrap": boot, "seed_spread_A_parity_lr1e-3": seeds,
           "test": test, "anchor_test": anchor, "label_in_slice": in_slice,
           "test_rows": int(len(S["y"])), "test_seqs": int(len(te)), "anchor_rows": int(len(A["y"])),
           "val_top1": tsum["val_top1"]}
    Tr.write_json(odir / "evaluate.json", res)
    log(json.dumps({"gate": gate, "gate_pair_bootstrap": boot["gate_pair"]}, indent=2))
    return res
