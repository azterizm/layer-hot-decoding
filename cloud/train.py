"""Single-pass training of every Phase 0 head side by side, and the pilot's anchor check.

Sequence batches go through the frozen teacher once. Head B trains on sub-batches of whole sequences.
The MLP heads draw random 2,048-row batches from a GPU shuffle buffer. Every head sees the same rows
in the same order; the h_1 and e arms of an architecture also share learning rate, seed and steps.
Validation top-1 is logged at 0.1M, 0.2M, 0.4M ... rows seen, which gives the data-scaling curve.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cloud import heads as Hd
from cloud.teacher import Batch, Teacher, plan_batches

A_BATCH = 2048
B_SEQS = 4
BUF_ROWS = 600_000
TEACHER_TOKENS = 49_152
VOCAB = 32_768
WARMUP_STEPS = 200  # fixed, so curve points past ~0.5M rows are not in warm-up
WEIGHT_DECAY = 0.01
CLIP = 1.0
VAL_PER_SEQ = 50
EVAL0 = 100_000
FIELDS = ("cur", "mid", "deep", "tok", "y")


def configs() -> list[dict]:
    c = []
    for arm in ("h1", "e", "deepmid"):
        for lr in (3e-4, 1e-3):
            c.append({"arch": "A-parity", "arm": arm, "lr": lr, "seed": 0})
    for arm in ("h1", "e"):
        for seed in (1, 2):
            c.append({"arch": "A-parity", "arm": arm, "lr": 1e-3, "seed": seed})
    for arch in ("A-large", "B"):
        for arm in ("h1", "e"):
            for lr in (1e-4, 3e-4):
                c.append({"arch": arch, "arm": arm, "lr": lr, "seed": 0})
    for x in c:
        x["name"] = f"{x['arch']}/{x['arm']}/lr{x['lr']:g}/s{x['seed']}"
    return c


def fname(name: str) -> str:
    return name.replace("/", "__")


def amp_ctx(device: str, on: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16) if on and device.startswith("cuda") else nullcontext()


def sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# ------------------------------------------------------------------ data on the volume

def load_tokens(root: Path) -> dict:
    """All generated shards: per sequence its prompt length, tokens (prompt + generated), loop flag."""
    out = {"id": [], "plen": [], "glen": [], "loop": [], "shard": [], "seq": []}
    for f in sorted((root / "tokens").glob("shard-*.npz")):
        with np.load(f) as z:
            z = {k: z[k] for k in ("id", "plen", "glen", "loop", "tokens")}
        shard = int(f.stem.split("-")[1])
        off = 0
        for k in range(len(z["id"])):
            T = int(z["plen"][k]) + int(z["glen"][k])
            out["seq"].append(z["tokens"][off:off + T].astype(np.int64))
            off += T
            out["id"].append(int(z["id"][k]))
            out["plen"].append(int(z["plen"][k]))
            out["glen"].append(int(z["glen"][k]))
            out["loop"].append(bool(z["loop"][k]))
            out["shard"].append(shard)
    for k in ("id", "plen", "glen", "shard"):
        out[k] = np.array(out[k], dtype=np.int64)
    out["loop"] = np.array(out["loop"], dtype=bool)
    return out


def load_meta(root: Path) -> dict[int, dict]:
    return {m["id"]: m for m in map(json.loads, (root / "corpus" / "meta.jsonl").read_text().splitlines())}


def usable(tok: dict, meta: dict, split: str) -> np.ndarray:
    """Indices into ``tok`` of sequences in ``split`` with at least one row and no repetition loop."""
    ok = [k for k in range(len(tok["id"]))
          if meta[int(tok["id"][k])]["split"] == split and not tok["loop"][k] and tok["glen"][k] >= 2]
    return np.array(ok, dtype=np.int64)


def pick_positions(plen: int, glen: int, k: int, seed: int) -> np.ndarray:
    """Up to k row positions i in [P, T-2] of one sequence, uniformly, sorted."""
    rows = np.arange(plen, plen + glen - 1)
    if len(rows) <= k:
        return rows
    return np.sort(np.random.default_rng(seed).choice(rows, size=k, replace=False))


def vocab_rows(tok: dict, idx: np.ndarray, n: int, vocab_size: int) -> np.ndarray:
    """The n most frequent row labels in the training split, using the generated next token as a stand-in
    for the label (they agree ~99 %). Most frequent first."""
    nxt = np.concatenate([tok["seq"][k][tok["plen"][k] + 1:] for k in idx])
    counts = np.bincount(nxt, minlength=vocab_size)[:vocab_size]
    order = np.argsort(-counts, kind="stable")
    return order[counts[order] > 0][:n]


# ------------------------------------------------------------------ shuffle buffer

class ShuffleBuffer:
    """Rows go in as teacher batches arrive and come out once each, in random order. The capacity grows
    with the stream (``frac`` of rows seen, between ``min_cap`` and ``cap``) so early curve points are
    not empty."""

    def __init__(self, cap: int, H: int, device: str, extra: int, frac: float = 0.25,
                 min_cap: int = 32_768, seed: int = 0):
        n = cap + extra
        self.t = {f: torch.empty(n, H, dtype=torch.bfloat16, device=device) for f in ("cur", "mid", "deep")}
        self.t["tok"] = torch.empty(n, dtype=torch.long, device=device)
        self.t["y"] = torch.empty(n, dtype=torch.long, device=device)
        self.alloc, self.cap, self.frac, self.min_cap = n, cap, frac, min_cap
        self.n = 0
        self.seen = 0
        self.device = device
        self.g = torch.Generator(device=device)
        self.g.manual_seed(seed)

    def push(self, rows: dict) -> dict | None:
        k = len(rows["y"])
        if self.n + k > self.alloc:
            raise RuntimeError("shuffle buffer overflow: raise `extra`")
        for f in FIELDS:
            self.t[f][self.n:self.n + k] = rows[f]
        self.n += k
        self.seen += k
        target = min(self.cap, max(self.min_cap, int(self.frac * self.seen)))
        return self.pop(self.n - target) if self.n > target else None

    def pop(self, k: int) -> dict:
        n = self.n
        sel = torch.randperm(n, device=self.device, generator=self.g)[:k]
        out = {f: self.t[f][sel] for f in FIELDS}
        taken = torch.zeros(n, dtype=torch.bool, device=self.device)
        taken[sel] = True
        holes = sel[sel < n - k]
        movers = torch.arange(n - k, n, device=self.device)[~taken[n - k:]]
        for f in FIELDS:
            self.t[f][holes] = self.t[f][movers]
        self.n = n - k
        return out

    def drain(self) -> dict | None:
        return self.pop(self.n) if self.n else None


def rows_dict(r) -> dict:
    return {"cur": r.cur, "mid": r.mid, "deep": r.deep, "tok": r.tok, "y": r.y}


def cat_rows(a: dict | None, b: dict | None) -> dict | None:
    if a is None:
        return b
    if b is None:
        return a
    return {f: torch.cat([a[f], b[f]]) for f in FIELDS}


# ------------------------------------------------------------------ evaluation chunks

@torch.no_grad()
def eval_chunks(teacher: Teacher, seqs: list, plens: list[int], picks: list[np.ndarray],
                chunk_seqs: int = 16, to_cpu: bool = False):
    """Teacher-force groups of sequences; ``picks[j]`` are the row positions of sequence j to score."""
    L1, Lm, Ld = teacher.capture
    for s in range(0, len(seqs), chunk_seqs):
        js = list(range(s, min(s + chunk_seqs, len(seqs))))
        b = teacher.run([seqs[j] for j in js], label_start=[plens[j] for j in js])
        sel = torch.zeros_like(b.label, dtype=torch.bool)
        for k, j in enumerate(js):
            sel[k, torch.as_tensor(picks[j], device=sel.device)] = True
        sel &= b.label >= 0
        ch = {"ids": b.ids, "cur": b.h[L1], "mid": b.h[Lm], "deep": b.h[Ld], "sel": sel, "js": js}
        if to_cpu:
            ch = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in ch.items()}
        ch["label"] = (b.label if not to_cpu else b.label.cpu())
        yield ch


@torch.no_grad()
def predict_chunk(teacher: Teacher, heads: dict, cfgs: dict, ch: dict, device: str, amp: bool) -> dict:
    """{name: (token id or -1, confidence)} for every selected row of one chunk, in (seq, pos) order."""
    ch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in ch.items()}
    s, t = ch["sel"].nonzero(as_tuple=True)
    rows = {"cur": ch["cur"][s, t], "emb": teacher.embed(ch["ids"][s, t]),
            "mid": ch["mid"][s, t - 1], "deep": ch["deep"][s, t - 1]}
    emb_full = None
    out = {}
    for name, head in heads.items():
        c = cfgs[name]
        with amp_ctx(device, amp):
            if c["arch"] == "B":
                if c["arm"] == "h1":
                    cur = ch["cur"]
                else:
                    emb_full = teacher.embed(ch["ids"]) if emb_full is None else emb_full
                    cur = emb_full
                logits = head(cur, ch["mid"], ch["deep"], (s, t))
            else:
                logits = head([rows[k] for k in Hd.ARMS[c["arm"]]])
        out[name] = head.guess(logits)
    return out


# ------------------------------------------------------------------ trainer

class Trainer:
    def __init__(self, teacher: Teacher, out: Hd.OutSlice, cfgs: list[dict], a_total: int, b_total: int,
                 device: str = "cuda", amp: bool = True, buf_rows: int = BUF_ROWS,
                 buf_extra: int = TEACHER_TOKENS + A_BATCH, arch_cfg: dict | None = None):
        self.teacher, self.out, self.device, self.amp = teacher, out, device, amp
        self.cfgs = {c["name"]: c for c in cfgs}
        H, eps = teacher.H, teacher.eps
        self.heads = {}
        self.opts = {}
        for c in cfgs:
            h = Hd.make(c["arch"], c["arm"], out, H, eps, c["seed"],
                        None if arch_cfg is None else arch_cfg[c["arch"]]).to(device)
            self.heads[c["name"]] = h
            self.opts[c["name"]] = torch.optim.AdamW(h.parameters(), lr=c["lr"], betas=(0.9, 0.999), eps=1e-8,
                                                     weight_decay=WEIGHT_DECAY, fused=device.startswith("cuda"))
        self.a_names = [n for n, c in self.cfgs.items() if c["arch"] != "B"]
        self.b_names = [n for n, c in self.cfgs.items() if c["arch"] == "B"]
        self.a_total, self.b_total = a_total, b_total
        self.a_step_n = 0
        self.b_step_n = 0
        self.a_rows = 0
        self.b_rows = 0
        self.loss = {n: torch.zeros((), device=device) for n in self.cfgs}
        self.loss_n = {n: 0 for n in self.cfgs}
        self.buf = ShuffleBuffer(buf_rows, H, device, extra=buf_extra) if self.a_names else None
        self.carry = None
        self.t_a = 0.0
        self.t_b = 0.0

    def _step(self, name: str, logits: torch.Tensor, y: torch.Tensor, step: int, total: int):
        c = self.cfgs[name]
        loss = F.cross_entropy(logits, y)
        opt = self.opts[name]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.heads[name].parameters(), CLIP)
        lr = Hd.lr_at(step, total, c["lr"], min(WARMUP_STEPS, total // 10))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.step()
        self.loss[name] += loss.detach()
        self.loss_n[name] += 1

    def a_batch(self, r: dict):
        y = self.out.map(r["y"])
        x = {"cur": r["cur"], "emb": self.teacher.embed(r["tok"]), "mid": r["mid"], "deep": r["deep"]}
        for name in self.a_names:
            with amp_ctx(self.device, self.amp):
                logits = self.heads[name]([x[k] for k in Hd.ARMS[self.cfgs[name]["arm"]]])
            self._step(name, logits, y, self.a_step_n, self.a_total)
        self.a_step_n += 1
        self.a_rows += len(y)

    def feed_a(self, rows: dict | None, final: bool = False):
        rows = cat_rows(self.carry, rows)
        self.carry = None
        if rows is None:
            return
        n = len(rows["y"])
        full = n if final else n // A_BATCH * A_BATCH
        for s in range(0, full, A_BATCH):
            self.a_batch({f: rows[f][s:s + A_BATCH] for f in FIELDS})
        if full < n:
            self.carry = {f: rows[f][full:].clone() for f in FIELDS}

    def b_batch(self, b: Batch, sl: slice, row_start: list[int]):
        L1, Lm, Ld = self.teacher.capture
        T = int(b.lens[sl].max())
        lab = b.label[sl, :T]
        pos = torch.arange(T, device=lab.device)
        rs = torch.as_tensor(row_start, device=lab.device)
        sel = ((lab >= 0) & (pos[None] >= rs[:, None])).nonzero(as_tuple=True)
        y = self.out.map(lab[sel])
        mid, deep = b.h[Lm][sl, :T], b.h[Ld][sl, :T]
        cur = {"h1": b.h[L1][sl, :T], "e": None}
        for name in self.b_names:
            arm = self.cfgs[name]["arm"]
            if cur[arm] is None:
                cur[arm] = self.teacher.embed(b.ids[sl, :T])
            with amp_ctx(self.device, self.amp):
                logits = self.heads[name](cur[arm], mid, deep, sel)
            self._step(name, logits, y, self.b_step_n, self.b_total)
        self.b_step_n += 1
        self.b_rows += len(y)

    def teacher_batch(self, b: Batch, plens: list[int]):
        t0 = time.time()
        if self.b_names:
            for s in range(0, len(plens), B_SEQS):
                self.b_batch(b, slice(s, s + B_SEQS), plens[s:s + B_SEQS])
        sync(self.device)
        t1 = time.time()
        if self.a_names:
            r = self.teacher.rows(b, plens)
            self.feed_a(self.buf.push(rows_dict(r)))
        sync(self.device)
        self.t_b += t1 - t0
        self.t_a += time.time() - t1

    def finish(self):
        if self.a_names:
            self.feed_a(self.buf.drain(), final=True)

    def mean_loss(self) -> dict:
        out = {n: (float(self.loss[n]) / self.loss_n[n] if self.loss_n[n] else None) for n in self.cfgs}
        for n in self.cfgs:
            self.loss[n].zero_()
            self.loss_n[n] = 0
        return out

    @torch.no_grad()
    def evaluate(self, chunks: list[dict]) -> dict:
        for h in self.heads.values():
            h.eval()
        hit = {n: 0 for n in self.cfgs}
        n_rows = 0
        for ch in chunks:
            pred = predict_chunk(self.teacher, self.heads, self.cfgs, ch, self.device, self.amp)
            y = ch["label"].to(self.device)[ch["sel"].to(self.device)]
            n_rows += len(y)
            for n, (tok, _) in pred.items():
                hit[n] += int((tok == y).sum())
        for h in self.heads.values():
            h.train()
        return {n: hit[n] / max(1, n_rows) for n in self.cfgs}

    def state(self) -> dict:
        return {"heads": {n: h.state_dict() for n, h in self.heads.items()},
                "opts": {n: o.state_dict() for n, o in self.opts.items()},
                "a_step_n": self.a_step_n, "b_step_n": self.b_step_n,
                "a_rows": self.a_rows, "b_rows": self.b_rows}

    def load_state(self, st: dict):
        for n, h in self.heads.items():
            h.load_state_dict(st["heads"][n])
            self.opts[n].load_state_dict(st["opts"][n])
        self.a_step_n, self.b_step_n = st["a_step_n"], st["b_step_n"]
        self.a_rows, self.b_rows = st["a_rows"], st["b_rows"]


def save_atomic(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


# ------------------------------------------------------------------ the main run

def run(root: Path, run_name: str = "main", meter=None, ckpt_every: float = 1800.0, max_seqs: int = 0,
        commit=None, log=print) -> dict:
    root = Path(root)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rdir = root / "train" / run_name
    tok = load_tokens(root)
    meta = load_meta(root)
    tr = usable(tok, meta, "train")
    va = usable(tok, meta, "val")
    tr = tr[np.random.default_rng(0).permutation(len(tr))]
    if max_seqs:
        tr = tr[:max_seqs]
    lengths = tok["plen"] + tok["glen"]
    rows_per = tok["glen"] - 1
    a_rows_total = int(rows_per[tr].sum())
    log(f"train seqs {len(tr)}, rows {a_rows_total}, val seqs {len(va)}, "
        f"loop-flagged {int(tok['loop'].sum())}/{len(tok['loop'])}")

    teacher = Teacher(device=device)
    rows = vocab_rows(tok, tr, VOCAB, teacher.vocab_size)
    out = Hd.OutSlice(teacher.norm_w, teacher.eps, teacher.lm_w, rows, teacher.vocab_size)
    batches = plan_batches(lengths, tr, TEACHER_TOKENS)
    a_total = math.ceil(a_rows_total / A_BATCH)
    b_total = sum(math.ceil(len(b) / B_SEQS) for b in batches)
    cfgs = configs()
    trainer = Trainer(teacher, out, cfgs, a_total, b_total, device=device)

    va = va[np.argsort(lengths[va], kind="stable")]  # less padding in the host-memory val chunks
    picks = [pick_positions(int(tok["plen"][k]), int(tok["glen"][k]), VAL_PER_SEQ, int(tok["id"][k])) for k in va]
    t0 = time.time()
    val = list(eval_chunks(teacher, [tok["seq"][k] for k in va], [int(tok["plen"][k]) for k in va], picks,
                           to_cpu=True))
    log(f"val chunks ready: {sum(int(c['sel'].sum()) for c in val)} rows ({time.time() - t0:.0f}s)")

    st = {"tb_next": 0, "stream_rows": 0, "next_eval": EVAL0, "curve": [], "lost_rows": 0,
          "agree": {}, "teacher_s": 0.0, "teacher_tokens": 0, "elapsed_before": 0.0}
    ck = rdir / "ckpt.pt"
    if ck.exists():
        saved = torch.load(ck, map_location=device, weights_only=False)
        trainer.load_state(saved["trainer"])
        st, lost = saved["st"], saved["in_flight_rows"]
        del saved
        torch.cuda.empty_cache()
        st["lost_rows"] += lost
        log(f"resumed at teacher batch {st['tb_next']}/{len(batches)}, lost {lost} buffered rows")
    else:
        (rdir / "curve.jsonl").unlink(missing_ok=True)

    write_json(rdir / "plan.json", {"configs": cfgs, "a_total_steps": a_total, "b_total_steps": b_total,
                                    "train_seqs": int(len(tr)), "train_rows": a_rows_total,
                                    "teacher_batches": len(batches), "vocab": int(len(rows)),
                                    "params": {n: Hd.n_params(h) for n, h in trainer.heads.items()},
                                    "a_batch": A_BATCH, "b_seqs": B_SEQS, "buffer_rows": BUF_ROWS,
                                    "teacher_tokens": TEACHER_TOKENS, "warmup_steps": WARMUP_STEPS})
    np.save(rdir / "vocab_rows.npy", rows)

    def log_eval(tag: str):
        ev = trainer.evaluate(val)
        rec = {"tag": tag, "stream_rows": st["stream_rows"], "a_rows": trainer.a_rows, "b_rows": trainer.b_rows,
               "a_steps": trainer.a_step_n, "b_steps": trainer.b_step_n,
               "elapsed_s": st["elapsed_before"] + time.time() - t_start, "val_top1": ev,
               "train_loss": trainer.mean_loss()}
        st["curve"].append(rec)
        with open(rdir / "curve.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        best = max(ev, key=ev.get)
        log(f"[eval {tag}] stream {st['stream_rows']:,} a_rows {trainer.a_rows:,} b_rows {trainer.b_rows:,} "
            f"best {best} {ev[best]:.4f} | " + " ".join(f"{n.split('/')[0]}/{n.split('/')[1]}:{v:.3f}"
                                                      for n, v in ev.items() if '/s0' in n))
        return ev

    t_start = time.time()
    t_ck = time.time()
    stopped = None
    for tb in range(st["tb_next"], len(batches)):
        idx = batches[tb]
        plens = [int(tok["plen"][k]) for k in idx]
        sync(device)
        t1 = time.time()
        b = teacher.run([tok["seq"][k] for k in idx], label_start=plens)
        sync(device)
        st["teacher_s"] += time.time() - t1
        st["teacher_tokens"] += int(b.lens.sum())
        # label == generated token, per shard
        T = b.ids.shape[1]
        nxt = torch.cat([b.ids[:, 1:], torch.zeros_like(b.ids[:, :1])], dim=1)
        ok = b.label >= 0
        same = ((b.label == nxt) & ok).sum(1).tolist()
        cnt = ok.sum(1).tolist()
        for k, s_, c_ in zip(idx, same, cnt):
            a = st["agree"].setdefault(str(int(tok["shard"][k])), [0, 0])
            a[0] += s_
            a[1] += c_
        trainer.teacher_batch(b, plens)
        st["stream_rows"] += int(sum(cnt))
        del b
        st["tb_next"] = tb + 1
        if st["stream_rows"] >= st["next_eval"]:
            log_eval(f"{st['next_eval'] / 1e6:g}M")
            while st["next_eval"] <= st["stream_rows"]:
                st["next_eval"] *= 2
        if tb % 50 == 0:
            el = time.time() - t_start
            log(f"tb {tb}/{len(batches)} stream {st['stream_rows']:,} rows "
                f"({st['stream_rows'] / max(el, 1):.0f}/s) teacher {st['teacher_tokens'] / max(st['teacher_s'], 1e-9):.0f} tok/s "
                f"a {trainer.t_a:.0f}s b {trainer.t_b:.0f}s teacher {st['teacher_s']:.0f}s "
                f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f}/{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB "
                f"spend ${meter.projected() if meter else 0:.2f}")
        over = meter is not None and not meter.guard()
        if time.time() - t_ck > ckpt_every or over:
            in_flight = trainer.buf.n + (0 if trainer.carry is None else len(trainer.carry["y"]))
            st_save = dict(st, elapsed_before=st["elapsed_before"] + time.time() - t_start)
            tc = time.time()
            save_atomic({"trainer": trainer.state(), "st": st_save, "in_flight_rows": in_flight}, ck)
            if commit:
                commit()
            log(f"checkpoint at tb {tb + 1} ({time.time() - tc:.0f}s)")
            t_ck = time.time()
        if over:
            stopped = "cost cap"
            log("stopping: the cost cap would be passed")
            break

    if stopped is None:
        trainer.finish()
    ev = log_eval("final" if stopped is None else "stopped")
    fin = rdir / "final"
    for n, h in trainer.heads.items():
        save_atomic({"cfg": trainer.cfgs[n], "state": h.state_dict()}, fin / f"{fname(n)}.pt")
    elapsed = st["elapsed_before"] + time.time() - t_start
    agree = {k: v[0] / max(1, v[1]) for k, v in st["agree"].items()}
    summary = {"stopped": stopped, "val_top1": ev, "a_rows": trainer.a_rows, "b_rows": trainer.b_rows,
               "a_steps": trainer.a_step_n, "b_steps": trainer.b_step_n, "stream_rows": st["stream_rows"],
               "lost_rows": st["lost_rows"], "elapsed_s": elapsed, "teacher_s": st["teacher_s"],
               "teacher_tok_s": st["teacher_tokens"] / max(st["teacher_s"], 1e-9),
               "a_s": trainer.t_a, "b_s": trainer.t_b,
               "label_eq_generated": {"overall": sum(v[0] for v in st["agree"].values())
                                      / max(1, sum(v[1] for v in st["agree"].values())),
                                      "min_shard": min(agree.values()) if agree else None, "by_shard": agree},
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "vocab_rows": int(len(rows)),
               "train_seqs": int(len(tr)), "val_rows": sum(int(c["sel"].sum()) for c in val)}
    write_json(rdir / "summary.json", summary)
    if ck.exists() and stopped is None:
        ck.unlink()
    log(json.dumps({k: v for k, v in summary.items() if k not in ("val_top1", "label_eq_generated")}))
    return summary


# ------------------------------------------------------------------ pilot: anchor check and dry run

def anchor_data(root: Path) -> list[dict]:
    return [json.loads(l) for l in (root / "anchor" / "gen.jsonl").read_text().splitlines()]


@torch.no_grad()
def anchor_features(teacher: Teacher, gen: list[dict], max_tokens: int = 32_768) -> dict:
    """Teacher-force the Mac run's sequences. Labels at stored positions P-1..T-2 (as collect.py), rows
    from P (as build_index). Returns rows on the GPU plus per-row metadata."""
    seqs = [np.array(g["prompt_ids"] + g["gen_ids"], dtype=np.int64) for g in gen]
    plens = [len(g["prompt_ids"]) for g in gen]
    lengths = np.array([len(s) for s in seqs])
    batches = plan_batches(lengths, np.arange(len(seqs)), max_tokens)
    R = {f: [] for f in FIELDS}
    R.update(seq=[], pos=[])
    stored_labels = {"train": [], "test": []}
    agree = [0, 0]
    t_s, n_tok = 0.0, 0
    prenorm = None
    for bi, idx in enumerate(batches):
        sync("cuda")
        t0 = time.time()
        b = teacher.run([seqs[j] for j in idx], label_start=[plens[j] - 1 for j in idx])
        sync("cuda")
        t_s += time.time() - t0
        n_tok += int(b.lens.sum())
        if bi == 0:
            prenorm = prenorm_check(teacher, b, seqs[idx[0]])
        for k, j in enumerate(idx):
            P, T = plens[j], len(seqs[j])
            lab = b.label[k, P - 1:T - 1]
            stored_labels[gen[j]["split"]].append(lab.cpu().numpy())
            agree[0] += int((lab == b.ids[k, P:T]).sum())
            agree[1] += len(lab)
        r = teacher.rows(b, [plens[j] for j in idx])
        for f, v in rows_dict(r).items():
            R[f].append(v)
        R["seq"].append(torch.as_tensor(np.asarray(idx), device="cuda")[r.seq])
        R["pos"].append(r.pos)
    out = {f: torch.cat(v) for f, v in R.items()}
    out["split"] = np.array([gen[j]["split"] for j in out["seq"].cpu().numpy()])
    out["kind"] = np.array([gen[j]["kind"] for j in out["seq"].cpu().numpy()])
    out["train_labels"] = np.concatenate(stored_labels["train"])
    out["agree"] = agree[0] / agree[1]
    out["teacher_tok_s"] = n_tok / t_s
    out["prenorm"] = prenorm
    return out


@torch.no_grad()
def prenorm_check(teacher: Teacher, b: Batch, seq: np.ndarray) -> dict:
    """lm_head(norm(h_36)) against the model's own logits, on one sequence run alone."""
    L = teacher.capture[-1]
    n = len(seq)
    ids = torch.as_tensor(seq[None], device=teacher.device)
    ref = teacher.model(input_ids=ids, use_cache=False).logits[0].float()
    hn = teacher.inner.norm(b.h[L][0, :n])
    mine = teacher.logits32(hn)
    return {"max_abs_norm_diff": float((hn.float() - b.hn[0, :n].float()).abs().max()),
            "max_abs_logit_diff": float((mine - ref).abs().max()),
            "argmax_agree": float((mine.argmax(-1) == ref.argmax(-1)).float().mean())}


class MLXAdamW:
    """mlx.optimizers.AdamW with its defaults (betas 0.9/0.999, eps 1e-8, weight decay 0.01, and no bias
    correction, which PyTorch's AdamW always applies)."""

    def __init__(self, params: list, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, wd: float = 0.01):
        self.params, self.lr, self.betas, self.eps, self.wd = params, lr, betas, eps, wd
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]

    def zero_grad(self, set_to_none: bool = True):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self, lr: float):
        b1, b2 = self.betas
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            p.mul_(1 - lr * self.wd)
            m.mul_(b1).add_(g, alpha=1 - b1)
            v.mul_(b2).addcmul_(g, g, value=1 - b2)
            p.sub_(lr * m / (v.sqrt() + self.eps))


def train_parity(teacher: Teacher, feats: dict, arm: str, vocab: int = 16_384, epochs: int = 6,
                 batch: int = 256, lr: float = 3e-4, log=print) -> dict:
    """The Mac Stage 1a run, ported: A-parity, fp32, AdamW(wd 0.01), cosine decay, no warm-up."""
    lab = feats["train_labels"]
    counts = np.bincount(lab, minlength=teacher.vocab_size)
    order = np.argsort(-counts, kind="stable")
    rows = order[counts[order] > 0][:vocab]
    out = Hd.OutSlice(teacher.norm_w, teacher.eps, teacher.lm_w, rows, teacher.vocab_size)
    head = Hd.make("A-parity", arm, out, teacher.H, teacher.eps, seed=0).cuda()
    opt = MLXAdamW(list(head.parameters()), lr=lr)
    tr = np.nonzero(feats["split"] == "train")[0]
    te = np.nonzero(feats["split"] == "test")[0]
    steps = epochs * math.ceil(len(tr) / batch)
    x = {"cur": feats["cur"], "mid": feats["mid"], "deep": feats["deep"]}
    y_all = out.map(feats["y"])
    keys = Hd.ARMS[arm]

    def inputs(sel):
        sel_t = torch.as_tensor(sel, device="cuda")
        d = {k: x[k][sel_t] for k in ("cur", "mid", "deep") if k in keys}
        if "emb" in keys:
            d["emb"] = teacher.embed(feats["tok"][sel_t])
        return [d[k] for k in keys], sel_t

    def evaluate():
        head.eval()
        toks = []
        with torch.no_grad():
            for i in range(0, len(te), 2048):
                xs, _ = inputs(te[i:i + 2048])
                toks.append(head.guess(head(xs))[0])
        head.train()
        tok = torch.cat(toks)
        return tok, (tok == feats["y"][torch.as_tensor(te, device="cuda")]).float()

    rng = np.random.default_rng(0)
    step = 0
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(tr)
        tot = torch.zeros((), device="cuda")
        for i in range(0, len(perm), batch):
            xs, sel_t = inputs(perm[i:i + batch])
            loss = F.cross_entropy(head(xs), y_all[sel_t])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(Hd.lr_at(step, steps, lr, 0))
            step += 1
            tot += loss.detach() * len(sel_t)
        _, hit = evaluate()
        hist.append({"epoch": ep, "train_loss": float(tot) / len(perm), "top1": float(hit.mean())})
        log(f"anchor {arm} epoch {ep} loss {hist[-1]['train_loss']:.3f} held-out top1 {hist[-1]['top1']:.4f} "
            f"({time.time() - t0:.0f}s)")
    _, hit = evaluate()
    hit = hit.cpu().numpy()
    kind = feats["kind"][te]
    return {"arm": arm, "train_positions": int(len(tr)), "test_positions": int(len(te)),
            "top1": float(hit.mean()), "history": hist,
            "by_kind": {k: float(hit[kind == k].mean()) for k in np.unique(kind)}}


def dry_run(teacher: Teacher, feats: dict, gen: list[dict], seconds: float = 150.0, log=print) -> dict:
    """All main-run heads on the anchor sequences, with the full-size buffer: step times and peak memory."""
    torch.backends.cuda.matmul.allow_tf32 = True
    seqs = [np.array(g["prompt_ids"] + g["gen_ids"], dtype=np.int64) for g in gen]
    plens = [len(g["prompt_ids"]) for g in gen]
    rows = np.argsort(-np.bincount(feats["train_labels"], minlength=teacher.vocab_size), kind="stable")[:VOCAB]
    out = Hd.OutSlice(teacher.norm_w, teacher.eps, teacher.lm_w, rows, teacher.vocab_size)
    cfgs = configs()
    torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(teacher, out, cfgs, a_total=25_000, b_total=30_000, device="cuda")
    # fill the buffer to its full size so peak memory is realistic
    for t in trainer.buf.t.values():
        t.zero_()
    trainer.buf.n = trainer.buf.cap
    trainer.buf.seen = 10 ** 9
    batches = plan_batches(np.array([len(s) for s in seqs]), np.arange(len(seqs)), TEACHER_TOKENS)
    t_teacher, n_tok, n_rows, n_seqs = 0.0, 0, 0, 0
    t0 = time.time()
    k = 0
    while time.time() - t0 < seconds:
        idx = batches[k % len(batches)]
        k += 1
        sync("cuda")
        t1 = time.time()
        b = teacher.run([seqs[j] for j in idx], label_start=[plens[j] for j in idx])
        sync("cuda")
        t_teacher += time.time() - t1
        n_tok += int(b.lens.sum())
        n_seqs += len(idx)
        n_rows += int(sum(len(seqs[j]) - plens[j] - 1 for j in idx))
        trainer.teacher_batch(b, [plens[j] for j in idx])
        del b
    # one worst-case teacher batch: the longest sequences the corpus allows (1024 prompt + 512 generated)
    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(0)
    n_w = TEACHER_TOKENS // 1536
    wseq = [torch.randint(0, 150_000, (1536,), generator=g).numpy() for _ in range(n_w)]
    for _ in range(2):
        b = teacher.run(wseq, label_start=[1024] * n_w)
        trainer.teacher_batch(b, [1024] * n_w)
        del b
    sync("cuda")
    worst = {"seqs": n_w, "len": 1536, "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
             "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
    log(f"worst-case batch: {worst}")
    tc = time.time()
    ev = trainer.evaluate(list(eval_chunks(teacher, seqs[:64], plens[:64],
                                           [np.arange(p, len(s) - 1) for s, p in zip(seqs[:64], plens[:64])],
                                           to_cpu=True)))
    t_eval = time.time() - tc
    res = {"seconds": time.time() - t0, "teacher_s": t_teacher, "teacher_tokens": n_tok,
           "teacher_tok_s": n_tok / t_teacher, "rows": n_rows, "seqs": n_seqs,
           "a_steps": trainer.a_step_n, "a_s": trainer.t_a, "a_s_per_step": trainer.t_a / max(1, trainer.a_step_n),
           "b_steps": trainer.b_step_n, "b_s": trainer.t_b, "b_s_per_step": trainer.t_b / max(1, trainer.b_step_n),
           "eval_s_64seqs": t_eval, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30,
           "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30, "worst_case": worst,
           "params": {n: Hd.n_params(h) for n, h in trainer.heads.items()},
           "val_check": {n: v for n, v in list(ev.items())[:3]}}
    del trainer
    torch.cuda.empty_cache()
    log(json.dumps({k: v for k, v in res.items() if k != "params"}))
    return res


def ckpt_speed(root: Path, gib: float = 4.0) -> dict:
    """Write and read back a ``gib`` GiB tensor on the volume to time checkpoint I/O."""
    x = torch.empty(int(gib * 2**30 / 4), dtype=torch.float32, device="cuda").normal_()
    p = root / "train" / "_speed.pt"
    t0 = time.time()
    save_atomic({"x": x}, p)
    w = time.time() - t0
    p.unlink()
    return {"gib": gib, "write_s": w, "write_gib_s": gib / w}


def pilot(root: Path, local_hash: str, log=print) -> dict:
    """Anchor check, then a dry run of the main loop for throughput and memory."""
    root = Path(root)
    from layerhot import prompts as P
    import hashlib

    items = P.build()
    h = hashlib.sha256(json.dumps([[it["id"], it["kind"], it["text"], it["split"], it["group"]]
                                   for it in items]).encode()).hexdigest()
    gen = anchor_data(root)
    assert all(items[g["id"]]["text"] == g["text"] and items[g["id"]]["split"] == g["split"] for g in gen)
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 parity with the MLX run
    teacher = Teacher(device="cuda")
    t0 = time.time()
    feats = anchor_features(teacher, gen)
    log(f"anchor features: {len(feats['y'])} rows, label==generated {feats['agree']:.4f}, "
        f"teacher {feats['teacher_tok_s']:.0f} tok/s, prenorm {feats['prenorm']} ({time.time() - t0:.0f}s)")
    res = {"prompts_hash": h, "prompts_hash_local": local_hash, "prompts_hash_match": h == local_hash,
           "label_eq_generated": feats["agree"], "prenorm": feats["prenorm"],
           "rows": int(len(feats["y"])), "anchor": {}}
    for arm in ("h1", "e", "deep"):
        res["anchor"][arm] = train_parity(teacher, feats, arm, log=log)
    write_json(root / "pilot" / "anchor.json", res)
    res["dry_run"] = dry_run(teacher, feats, gen, log=log)
    res["ckpt_io"] = ckpt_speed(root)
    write_json(root / "pilot" / "anchor.json", res)
    return res
