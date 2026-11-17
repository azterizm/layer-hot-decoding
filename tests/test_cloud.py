"""CPU tests for cloud/ (Phase 0). They need torch and transformers, so they run inside Modal:
``modal run -m cloud.app::step1_selftest``. A tiny random Qwen3 stands in for the 4B model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cloud import evaluate as Ev  # noqa: E402
from cloud import gen as G  # noqa: E402
from cloud import heads as Hd  # noqa: E402
from cloud import ledger  # noqa: E402
from cloud import train as Tr  # noqa: E402
from cloud.teacher import Teacher, plan_batches  # noqa: E402

V, H, NL = 97, 64, 4
CAP = (1, 2, 4)


@pytest.fixture(scope="module")
def teacher():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=V, hidden_size=H, intermediate_size=128, num_hidden_layers=NL,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                      tie_word_embeddings=True, rms_norm_eps=1e-6)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).float()
    return Teacher(model=m, capture=CAP, device="cpu", dtype=torch.float32)


def seqs_and_plens(n=5, seed=0):
    rng = np.random.default_rng(seed)
    plens = [int(rng.integers(2, 9)) for _ in range(n)]
    seqs = [rng.integers(0, V, size=p + int(rng.integers(2, 20))).astype(np.int64) for p in plens]
    return seqs, plens


def build_index(prompt: np.ndarray):
    """layerhot.deep_head.build_index (that file imports MLX)."""
    first = np.r_[True, prompt[1:] != prompt[:-1]]
    cur = np.nonzero(~first)[0]
    return cur, cur - 1


def test_hooks_are_pre_norm_residuals(teacher):
    seqs, plens = seqs_and_plens(1)
    ids = torch.as_tensor(seqs[0][None])
    b = teacher.run(seqs, label_start=plens)
    hs = teacher.model(input_ids=ids, output_hidden_states=True).hidden_states
    for L in CAP[:-1]:
        assert torch.allclose(b.h[L][0], hs[L][0], atol=1e-5)
    # the last entry of hidden_states is after the final norm, the hook is before it
    assert torch.allclose(teacher.inner.norm(b.h[NL][0]), hs[NL][0], atol=1e-5)
    assert not torch.allclose(b.h[NL][0], hs[NL][0], atol=1e-3)


def test_prenorm_logits_match_model(teacher):
    seqs, plens = seqs_and_plens(1, seed=1)
    b = teacher.run(seqs, label_start=plens)
    ref = teacher.model(input_ids=torch.as_tensor(seqs[0][None])).logits[0]
    mine = teacher.logits32(teacher.inner.norm(b.h[NL][0]))
    assert torch.allclose(mine, ref, atol=1e-4)
    P, T = plens[0], len(seqs[0])
    assert torch.equal(b.label[0, P:T - 1], ref[P:T - 1].argmax(-1))
    assert (b.label[0, :P] == -1).all() and b.label[0, T - 1] == -1


def test_rows_match_build_index(teacher):
    seqs, plens = seqs_and_plens(6, seed=2)
    # collect.py stores positions P-1 .. T-2 per sequence; build_index pairs (i, i-1) within a sequence
    prompt, where = [], []
    for k, (s, P) in enumerate(zip(seqs, plens)):
        for j in range(P - 1, len(s) - 1):
            prompt.append(k)
            where.append((k, j))
    cur, prev = build_index(np.array(prompt))
    b = teacher.run(seqs, label_start=[p - 1 for p in plens])
    r = teacher.rows(b, plens)
    got = list(zip(r.seq.tolist(), r.pos.tolist()))
    assert got == [where[c] for c in cur]
    assert all(where[p] == (k, i - 1) for p, (k, i) in zip(prev, got))
    L1, Lm, Ld = CAP
    for n, (k, i) in enumerate(got):
        assert torch.equal(r.cur[n], b.h[L1][k, i])
        assert torch.equal(r.deep[n], b.h[Ld][k, i - 1])
        assert torch.equal(r.mid[n], b.h[Lm][k, i - 1])
        assert r.tok[n] == int(seqs[k][i])


def test_padding_does_not_change_states(teacher):
    seqs, plens = seqs_and_plens(4, seed=3)
    both = teacher.run(seqs, label_start=plens)
    for k, (s, P) in enumerate(zip(seqs, plens)):
        one = teacher.run([s], label_start=[P])
        T = len(s)
        for L in CAP:
            assert torch.allclose(one.h[L][0], both.h[L][k, :T], atol=1e-4)
        assert torch.equal(one.label[0], both.label[k, :T])


def make_out(teacher, n_rows=50):
    return Hd.OutSlice(teacher.norm_w, teacher.eps, teacher.lm_w, np.arange(n_rows), teacher.vocab_size)


def test_mlp_head_identity_start(teacher):
    out = make_out(teacher)
    for blocks in (1, 4):
        h = Hd.MLPHead(out, H, 1e-6, n_in=3, inner=2 * H, blocks=blocks)
        xs = [torch.randn(7, H) for _ in range(3)]
        base = h.fc(torch.cat([n(x) for n, x in zip(h.norms, xs)], dim=-1))
        assert torch.equal(h.body(xs), base)
        lg = h(xs)
        assert lg.shape == (7, out.V + 1)
        assert torch.allclose(lg[:, -1], torch.full((7,), -2.0))  # abstain logit starts at b_abs


def test_arms_share_initial_weights(teacher):
    out = make_out(teacher)
    a = Hd.make("A-parity", "h1", out, H, 1e-6, seed=0)
    b = Hd.make("A-parity", "e", out, H, 1e-6, seed=0)
    c = Hd.make("A-parity", "h1", out, H, 1e-6, seed=1)
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert not torch.equal(sa["fc.weight"], sc["fc.weight"])


SMALL = {"A-parity": {"kind": "mlp", "blocks": 1, "inner_mult": 2},
         "A-large": {"kind": "mlp", "blocks": 2, "inner_mult": 2},
         "B": {"kind": "x", "D": 32, "n_heads": 4, "mlp": 48}}


def test_head_b_no_future_leakage(teacher):
    out = make_out(teacher)
    torch.manual_seed(0)
    h = Hd.XHead(out, H, 1e-6, D=32, n_heads=4, mlp=48)
    for p in h.parameters():  # zero-initialised output projections would hide leaks
        torch.nn.init.normal_(p, std=0.2)
    B, T = 2, 12
    cur, mid, deep = (torch.randn(B, T, H) for _ in range(3))
    base = h.body(cur, mid, deep)
    for i in (0, 3, T - 2):
        c2, m2, d2 = cur.clone(), mid.clone(), deep.clone()
        c2[:, i + 1:] += 5
        m2[:, i:] += 5  # the memory at i itself is not allowed
        d2[:, i:] += 5
        y = h.body(c2, m2, d2)
        assert torch.allclose(y[:, :i + 1], base[:, :i + 1], atol=1e-5), i
    # the previous position's memory does reach position i
    m3 = mid.clone()
    m3[:, 4] += 5
    assert not torch.allclose(h.body(cur, m3, deep)[:, 5], base[:, 5], atol=1e-3)
    sel = (torch.tensor([0, 1]), torch.tensor([2, 7]))
    assert torch.allclose(h(cur, mid, deep, sel), h.logits(base[sel]), atol=1e-5)


def test_shuffle_buffer_emits_each_row_once():
    buf = Tr.ShuffleBuffer(cap=100, H=4, device="cpu", extra=64, frac=0.25, min_cap=16)
    seen, n = [], 0
    rng = np.random.default_rng(0)
    for _ in range(40):
        k = int(rng.integers(1, 60))
        rows = {f: torch.zeros(k, 4, dtype=torch.bfloat16) for f in ("cur", "mid", "deep")}
        rows["tok"] = torch.arange(n, n + k)
        rows["y"] = torch.arange(n, n + k)
        rows["cur"][:, 0] = rows["tok"].to(torch.bfloat16)
        n += k
        out = buf.push(rows)
        assert buf.n <= buf.cap
        if out is not None:
            assert torch.equal(out["tok"], out["y"])
            seen += out["tok"].tolist()
    seen += buf.drain()["tok"].tolist()
    assert sorted(seen) == list(range(n))
    assert seen != list(range(n))  # shuffled


def test_trainer_smoke(teacher):
    seqs, plens = seqs_and_plens(12, seed=4)
    out = make_out(teacher, 60)
    cfgs = [c for c in Tr.configs() if c["seed"] == 0 and c["lr"] in (3e-4, 1e-4) or c["arch"] == "B"]
    tr = Tr.Trainer(teacher, out, cfgs, a_total=50, b_total=50, device="cpu", amp=False, buf_rows=64,
                    buf_extra=512, arch_cfg=SMALL)
    tr.buf.min_cap = 16
    batches = plan_batches(np.array([len(s) for s in seqs]), np.arange(len(seqs)), max_tokens=80)
    rows = 0
    for idx in batches:
        b = teacher.run([seqs[j] for j in idx], label_start=[plens[j] for j in idx])
        rows += int((b.label >= 0).sum())
        tr.teacher_batch(b, [plens[j] for j in idx])
    tr.finish()
    assert tr.a_rows == rows and tr.b_rows == rows
    assert tr.buf.n == 0 and tr.carry is None
    chunks = list(Tr.eval_chunks(teacher, seqs, plens, [np.arange(p, len(s) - 1) for s, p in zip(seqs, plens)],
                                 chunk_seqs=5, to_cpu=True))
    ev = tr.evaluate(chunks)
    assert set(ev) == {c["name"] for c in cfgs} and all(0 <= v <= 1 for v in ev.values())
    loss = tr.mean_loss()
    assert all(v is not None and np.isfinite(v) for v in loss.values())


def test_plan_batches_cover_once_within_budget():
    rng = np.random.default_rng(0)
    lengths = rng.integers(5, 300, size=500)
    order = rng.permutation(500)
    bs = plan_batches(lengths, order, max_tokens=2048, window=128)
    flat = np.concatenate(bs)
    assert sorted(flat.tolist()) == list(range(500))
    assert all(lengths[b].max() * len(b) <= 2048 or len(b) == 1 for b in bs)


def test_lr_schedule():
    assert Hd.lr_at(0, 100, 1.0, 0) == 1.0
    assert Hd.lr_at(100, 100, 1.0, 0) < 1e-12
    assert Hd.lr_at(0, 100, 1.0, 10) == pytest.approx(0.1)
    assert Hd.lr_at(10, 100, 1.0, 10) == 1.0


def test_pick_positions():
    p = Tr.pick_positions(10, 30, 5, seed=1)
    assert len(p) == 5 and p.min() >= 10 and p.max() <= 10 + 30 - 2
    assert np.array_equal(Tr.pick_positions(10, 4, 50, 0), np.arange(10, 13))


def test_is_loop():
    rng = np.random.default_rng(0)
    assert G.is_loop(list(rng.integers(0, 1000, 100)) + [5, 6, 7] * 100)
    assert not G.is_loop(list(rng.integers(0, 1000, 400)))
    assert not G.is_loop([1, 2, 3] * 10)  # too short to judge


def test_bootstrap_and_gate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(300), 20)
    a = (rng.random(6000) < 0.5).astype(float)
    r = Ev.bootstrap_diff(a, a, groups, n=200)
    assert r["points"] == 0 and r["ci95"] == [0.0, 0.0]
    b = (rng.random(6000) < 0.3).astype(float)
    r = Ev.bootstrap_diff(a, b, groups, n=500)
    assert r["ci95"][0] < r["points"] < r["ci95"][1] and r["ci95"][0] > 10

    cfgs = Tr.configs()
    val = {c["name"]: 0.3 for c in cfgs}
    val["B/e/lr0.0003/s0"] = 0.6
    val["B/h1/lr0.0001/s0"] = 0.5
    ch = Ev.choose(val, cfgs)
    assert ch["best"] == "B/e/lr0.0003/s0" and ch["arch"] == "B" and ch["lr"] == 3e-4
    test = {n: {"top1": 0.5, "by_kind": {"code": 0.5}} for n in val}
    test[ch["best"]] = {"top1": 0.66, "by_kind": {"code": 0.7}}
    test[ch["h1"]] = {"top1": 0.70, "by_kind": {"code": 0.7}}
    d = Ev.decide(ch, test)
    assert d["model_gate"] == "pass" and d["hypothesis_gate"] == "pass" and d["revive_layer_hot"]
    test[ch["h1"]] = {"top1": 0.675, "by_kind": {"code": 0.7}}
    assert Ev.decide(ch, test)["hypothesis_gate"] == "small effect (fails)"
    test[ch["h1"]] = {"top1": 0.665, "by_kind": {"code": 0.7}}
    assert Ev.decide(ch, test)["hypothesis_gate"] == "disproven"


def test_ledger(tmp_path):
    assert ledger.rate_per_s("H100", 0, 0) * 3600 == pytest.approx(3.95)
    m = ledger.Meter(tmp_path, "x", "H100", 4, 16, heartbeat_s=3600)
    m.close()
    recs = ledger.read_all(tmp_path)
    assert len(recs) == 1 and recs[0]["usd"] > 0
    with pytest.raises(ledger.OverCap):
        ledger.check(39.0, 2.0, "x")
    ledger.check(10.0, 2.0, "x")


def test_configs_pair_arms():
    cfgs = Tr.configs()
    names = {c["name"] for c in cfgs}
    assert len(names) == len(cfgs) == 18
    for c in cfgs:
        if c["arm"] == "h1":
            assert c["name"].replace("/h1/", "/e/") in names
