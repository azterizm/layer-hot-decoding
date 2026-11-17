"""Prompt corpus for the cloud job: chat, code, math and prose prompts from pinned ungated datasets.

fetch() downloads the pinned files, build() makes the items, encode() tokenizes one item the same way
as layerhot.prompts.encode, summary() describes a corpus. Everything is deterministic from the seed.
The split is by group (conversation, code problem, math question, wikitext article).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import re
from collections import Counter

from layerhot import prompts

REVISIONS = {
    "HuggingFaceH4/ultrachat_200k": "8049631c405ae6576f93f445c6b8166f76f5505a",
    "OpenAssistant/oasst1": "fdf72ae0827c1cda404aff25b6603abec9e3399b",
    "ise-uiuc/Magicoder-OSS-Instruct-75K": "5f839b1f368a76b161028bb9edff055db34022b2",
    "m-a-p/CodeFeedback-Filtered-Instruction": "a08c213a9748c66c15d0225814be80a2e77adf4a",
    "meta-math/MetaMathQA": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
    "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
}
FILES = {
    "HuggingFaceH4/ultrachat_200k": ["data/train_sft-00000-of-00003-a3ecf92756993583.parquet"],
    "OpenAssistant/oasst1": ["data/train-00000-of-00001-b42a775f407cee45.parquet"],
    "ise-uiuc/Magicoder-OSS-Instruct-75K": ["data-oss_instruct-decontaminated.jsonl"],
    "m-a-p/CodeFeedback-Filtered-Instruction": ["CodeFeedback-Filtered-Instruction.jsonl"],
    "meta-math/MetaMathQA": ["MetaMathQA-395K.json"],
    "Salesforce/wikitext": ["wikitext-103-raw-v1/train-00000-of-00002.parquet",
                            "wikitext-103-raw-v1/train-00001-of-00002.parquet"],
}
KIND_SHARE = {"chat": 0.35, "code": 0.35, "prose": 0.20, "math": 0.10}
CODE_SHARE = {"python": 0.45, "jsts": 0.20, "rust": 0.15, "other": 0.20}
NOTES: list[str] = []  # what the last build() had to adjust (short buckets, fills)


def fetch(hf_home: str | None = None) -> dict[str, list[str]]:
    from huggingface_hub import hf_hub_download

    cache = os.path.join(hf_home, "hub") if hf_home else None
    return {repo: [hf_hub_download(repo, f, repo_type="dataset", revision=REVISIONS[repo], cache_dir=cache)
                   for f in files]
            for repo, files in FILES.items()}


# ---- small pieces (tested without the datasets)

def norm(s: str) -> str:
    return " ".join(s.split())


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def ultrachat_cut(messages: list[dict], rng: random.Random) -> list[dict]:
    """Cut at user turn k, k uniform in 1..min(3, user turns); keep everything up to and including it."""
    users = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if not users:
        return []
    end = users[rng.randint(1, min(3, len(users))) - 1]
    return [{"role": m["role"], "content": m["content"]} for m in messages[: end + 1]]


_TITLE = re.compile(r"^= [^=].*[^=] =$|^= [^=] =$")


def wikitext_paragraphs(lines, exclude=frozenset()) -> dict[int, list[str]]:
    """article index -> 60-word prompts of its long paragraphs (same rules as prompts._wikitext_paragraphs).

    A top-level title looks like ' = Title = '; ' = = Sub = = ' is a subheading. Lines before the
    first title belong to article -1."""
    out: dict[int, list[str]] = {}
    art = -1
    for t in lines:
        s = t.strip()
        if _TITLE.match(s):
            art += 1
        elif len(t) > 600 and not s.startswith("="):
            p = " ".join(prompts._detok(s).split()[:60])
            if p not in exclude:
                out.setdefault(art, []).append(p)
    return out


def code_bucket(lang: str) -> str:
    l = (lang or "").strip().lower()
    if l == "python":
        return "python"
    if l in ("javascript", "typescript", "js", "ts", "jsx", "tsx"):
        return "jsts"
    if l == "rust":
        return "rust"
    return "other"


def counts_by_share(n: int, shares: dict[str, float]) -> dict[str, int]:
    """Integer counts that sum to n, by largest remainder."""
    raw = {k: n * v for k, v in shares.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), k))[: n - sum(out.values())]:
        out[k] += 1
    return out


def split_items(items: list[dict], n_val: int, n_test: int, seed: int = 0) -> None:
    """Set item['split']. Whole groups only; per-kind counts follow KIND_SHARE; test first, then val."""
    rng = random.Random(seed + 100)
    by_kind: dict[str, dict[str, list[dict]]] = {}
    for it in items:
        it["split"] = "train"
        by_kind.setdefault(it["kind"], {}).setdefault(it["group"], []).append(it)
    for name, n in (("test", n_test), ("val", n_val)):
        want = counts_by_share(n, KIND_SHARE)
        for kind in sorted(by_kind):
            groups = sorted(g for g, its in by_kind[kind].items() if its[0]["split"] == "train")
            rng.shuffle(groups)
            got = 0
            for g in groups:
                its = by_kind[kind][g]
                if got + len(its) <= want.get(kind, 0):
                    for it in its:
                        it["split"] = name
                    got += len(its)
                if got == want.get(kind, 0):
                    break
            if got < want.get(kind, 0):
                NOTES.append(f"{name} {kind}: {got} of {want[kind]}")


def encode(tokenizer, item: dict) -> list[int]:
    if item["kind"] == "prose":
        return tokenizer.encode(item["text"])
    text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def lazy_keep(tokenizer, cands, max_tokens: int):
    """Encode candidates one by one and yield the ones with 1..max_tokens tokens (prompt_ids set).

    Used with take(), so only as many candidates are tokenized as the target needs."""
    for it in cands:
        ids = encode(tokenizer, it)
        if 0 < len(ids) <= max_tokens:
            it["prompt_ids"] = ids
            yield it


def dedupe(cands: list[dict], key) -> list[dict]:
    seen, out = set(), []
    for it in cands:
        k = norm(key(it))
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def take(it, n: int) -> list[dict]:
    return list(itertools.islice(it, max(n, 0)))


# ---- loaders: one list of candidate items per source (no tokenizing here)

def _item(kind, source, group, **kw) -> dict:
    return {"kind": kind, "source": source, "group": group, **kw}


def load_ultrachat(path: str, rng: random.Random) -> list[dict]:
    import pyarrow.parquet as pq

    d = pq.read_table(path, columns=["prompt_id", "messages"]).to_pydict()
    out = []
    for pid, msgs in zip(d["prompt_id"], d["messages"]):
        cut = ultrachat_cut(msgs, rng)
        if cut:
            out.append(_item("chat", "ultrachat", f"ultrachat:{pid}", messages=cut))
    return out


def load_oasst(path: str) -> list[dict]:
    import pyarrow.parquet as pq

    cols = ["parent_id", "role", "lang", "deleted", "text", "message_tree_id"]
    d = pq.read_table(path, columns=cols).to_pydict()
    return [_item("chat", "oasst", f"oasst:{tree}", messages=[{"role": "user", "content": text}])
            for pid, role, lang, deleted, text, tree in zip(*(d[c] for c in cols))
            if pid is None and role == "prompter" and lang == "en" and not deleted and text]


def _jsonl(path: str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_code(magicoder: str, codefeedback: str) -> dict[str, list[dict]]:
    out = {"magicoder": [], "codefeedback": []}
    for src, path, field in (("magicoder", magicoder, "problem"), ("codefeedback", codefeedback, "query")):
        for r in _jsonl(path):
            if r.get(field):
                out[src].append(_item("code", src, "", messages=[{"role": "user", "content": r[field]}],
                                      lang=(r.get("lang") or "").lower()))
    return out


def load_math(path: str) -> list[dict]:
    with open(path) as f:
        rows = json.load(f)
    return [_item("math", "metamath", "metamath:" + sha1(r["original_question"]),
                  messages=[{"role": "user", "content": r["query"]}])
            for r in rows if r.get("query")]


def load_prose(paths: list[str], rng: random.Random) -> list[dict]:
    import pyarrow.parquet as pq

    anchors = set(prompts._wikitext_paragraphs(150, 0))
    lines = itertools.chain.from_iterable(pq.read_table(p, columns=["text"]).column("text").to_pylist()
                                          for p in paths)
    arts = wikitext_paragraphs(lines, anchors)
    return [_item("prose", "wikitext", f"wikitext:{a}", text=rng.choice(arts[a])) for a in sorted(arts)]


# ---- build

def _text(it: dict) -> str:
    return "\n".join(m["content"] for m in it["messages"])


def _code_pool(tokenizer, cands: dict[str, list[dict]], rng: random.Random, max_tokens: int, n_code: int):
    """Code items: per language bucket, half from each source; short sources and buckets are filled from the rest."""
    its = {}
    for src, lst in cands.items():
        lst = dedupe(lst, _text)
        for it in lst:
            it["group"] = "code:" + sha1(norm(_text(it)))
        rng.shuffle(lst)
        its[src] = {b: lazy_keep(tokenizer, [x for x in lst if code_bucket(x["lang"]) == b], max_tokens)
                    for b in CODE_SHARE}
    want = counts_by_share(n_code, CODE_SHARE)
    got: dict[str, list[dict]] = {}
    for b in CODE_SHARE:
        a, c = sorted(its)  # codefeedback, magicoder
        half = want[b] // 2
        x = take(its[a][b], want[b] - half)
        y = take(its[c][b], want[b] - len(x))
        x += take(its[a][b], want[b] - len(x) - len(y))
        got[b] = x + y
        if len(got[b]) < want[b]:
            NOTES.append(f"code {b}: {len(got[b])} of {want[b]} available")
    short = n_code - sum(map(len, got.values()))
    for b in ("python", "other", "jsts", "rust"):  # fill from the bulk first
        for src in sorted(its):
            if short > 0:
                more = take(its[src][b], short)
                got[b] += more
                short -= len(more)
                if more:
                    NOTES.append(f"code {b}: +{len(more)} to fill the code share")
    return [it for b in CODE_SHARE for it in got[b]]


def build(tokenizer, seed: int = 0, n_total: int = 115_000, n_val: int = 1000, n_test: int = 2000,
          max_prompt_tokens: int = 1024) -> list[dict]:
    del NOTES[:]
    paths = fetch()
    want = counts_by_share(n_total, KIND_SHARE)
    rngs = [random.Random(seed + k) for k in range(1, 8)]

    chat_cands = dedupe(load_oasst(paths["OpenAssistant/oasst1"][0]), _text)
    uc = dedupe(load_ultrachat(paths["HuggingFaceH4/ultrachat_200k"][0], rngs[0]), _text)
    rngs[1].shuffle(uc)
    oasst = take(lazy_keep(tokenizer, chat_cands, max_prompt_tokens), want["chat"])
    chat = oasst + take(lazy_keep(tokenizer, uc, max_prompt_tokens), want["chat"] - len(oasst))
    NOTES.append(f"chat: {len(oasst)} oasst English roots kept, {len(chat) - len(oasst)} ultrachat")

    code = load_code(paths["ise-uiuc/Magicoder-OSS-Instruct-75K"][0],
                     paths["m-a-p/CodeFeedback-Filtered-Instruction"][0])
    code = _code_pool(tokenizer, code, rngs[2], max_prompt_tokens, want["code"])

    math_c = dedupe(load_math(paths["meta-math/MetaMathQA"][0]), _text)
    rngs[3].shuffle(math_c)
    math_ = take(lazy_keep(tokenizer, math_c, max_prompt_tokens), want["math"])

    prose_c = load_prose(paths["Salesforce/wikitext"], rngs[4])
    prose_c = dedupe(prose_c, lambda it: it["text"])
    rngs[5].shuffle(prose_c)
    prose = take(lazy_keep(tokenizer, prose_c, max_prompt_tokens), want["prose"])

    for kind, got in (("chat", chat), ("code", code), ("math", math_), ("prose", prose)):
        if len(got) < want[kind]:
            NOTES.append(f"{kind}: {len(got)} of {want[kind]} (short)")
    items = chat + code + math_ + prose
    split_items(items, n_val, n_test, seed)
    random.Random(seed).shuffle(items)
    for i, it in enumerate(items):
        it["id"] = i
    for n in NOTES:
        print("corpus:", n)
    return items


def summary(items: list[dict]) -> dict:
    def stats(xs):
        xs = sorted(xs)
        if not xs:
            return {}
        return {"n": len(xs), "min": xs[0], "mean": round(sum(xs) / len(xs), 1), "median": xs[len(xs) // 2],
                "p95": xs[min(len(xs) - 1, math.ceil(0.95 * len(xs)) - 1)], "max": xs[-1]}

    n = len(items)
    kinds = Counter(it["kind"] for it in items)
    return {
        "n": n,
        "kind": dict(kinds),
        "kind_share": {k: round(v / n, 4) for k, v in kinds.items()} if n else {},
        "source": dict(Counter(it["source"] for it in items)),
        "split": dict(Counter(it["split"] for it in items)),
        "kind_by_split": {s: dict(Counter(it["kind"] for it in items if it["split"] == s))
                          for s in ("train", "val", "test")},
        "code_lang": dict(Counter(code_bucket(it.get("lang", "")) for it in items if it["kind"] == "code")),
        "prompt_tokens": stats([len(it["prompt_ids"]) for it in items]),
        "prompt_tokens_by_kind": {k: stats([len(it["prompt_ids"]) for it in items if it["kind"] == k])
                                  for k in sorted(kinds)},
    }
