"""Checks for cloud/corpus.py that do not need the dataset files.

Env: LAYERHOT_TOKENIZER (default: pinned Qwen/Qwen3-4B; locally e.g. mlx-community/Qwen3-4B-4bit).
"""

from __future__ import annotations

import os
import random
from collections import Counter

import pytest

from cloud import corpus
from layerhot import prompts

DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"
DEFAULT_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    name = os.environ.get("LAYERHOT_TOKENIZER", DEFAULT_TOKENIZER)
    rev = DEFAULT_REVISION if name == DEFAULT_TOKENIZER else None
    return AutoTokenizer.from_pretrained(name, revision=rev)


def test_encode_matches_prompts_encode(tok):
    text = "Explain how tides work in simple terms."
    item = {"kind": "chat", "messages": [{"role": "user", "content": text}]}
    assert corpus.encode(tok, item) == prompts.encode(tok, {"kind": "chat", "text": text})
    prose = {"kind": "prose", "text": "The river runs south for most of its length."}
    assert corpus.encode(tok, prose) == prompts.encode(tok, prose)


def test_multi_turn_ends_with_generation_prompt(tok):
    msgs = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is 2+2?"}]
    ids = corpus.encode(tok, {"kind": "chat", "messages": msgs})
    text = tok.decode(ids)
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert text.count("<|im_start|>user") == 2 and "Hello!" in text


def _items():
    rng = random.Random(1)
    items = []
    n = 0
    for kind, count in (("chat", 700), ("code", 700), ("prose", 400), ("math", 200)):
        for i in range(count * 6):
            # math groups hold 1-3 items, the rest are single
            g = f"{kind}:{n}"
            size = rng.randint(1, 3) if kind == "math" else 1
            for _ in range(size):
                items.append({"kind": kind, "group": g, "source": kind})
            n += 1
    return items


def test_split_by_group_and_counts():
    items = _items()
    corpus.split_items(items, n_val=1000, n_test=2000, seed=3)
    splits = {}
    for it in items:
        splits.setdefault(it["group"], set()).add(it["split"])
    assert all(len(s) == 1 for s in splits.values())
    for split, n in (("test", 2000), ("val", 1000)):
        c = Counter(it["kind"] for it in items if it["split"] == split)
        assert c == corpus.counts_by_share(n, corpus.KIND_SHARE)
    assert Counter(it["kind"] for it in items if it["split"] == "test") == \
        {"chat": 700, "code": 700, "prose": 400, "math": 200}
    # deterministic
    again = _items()
    corpus.split_items(again, n_val=1000, n_test=2000, seed=3)
    assert [i["split"] for i in again] == [i["split"] for i in items]


def test_wikitext_article_grouping():
    long1 = " ".join(f"word{i}" for i in range(200))
    long2 = " ".join(f"other{i}" for i in range(200))
    lines = [
        " \n",
        " = Alpha = \n",
        " " + long1 + " \n",
        " = = Sub heading = = \n",
        " " + long2 + " \n",
        " short line \n",
        " = Beta = \n",
        " = = = Deep = = = \n",
        " " + long2 + " \n",
        " = Gamma = \n",
        " short \n",
    ]
    arts = corpus.wikitext_paragraphs(lines)
    assert sorted(arts) == [0, 1]
    assert len(arts[0]) == 2 and len(arts[1]) == 1
    assert all(len(p.split()) == 60 for ps in arts.values() for p in ps)
    ex = corpus.wikitext_paragraphs(lines, exclude={arts[1][0]})
    assert sorted(ex) == [0]


def test_ultrachat_turn_cut():
    msgs = []
    for i in range(4):
        msgs += [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
    seen = Counter()
    rng = random.Random(0)
    for _ in range(300):
        cut = corpus.ultrachat_cut(msgs, rng)
        assert cut[-1]["role"] == "user"
        assert cut == msgs[: len(cut)]
        seen[sum(m["role"] == "user" for m in cut)] += 1
    assert sorted(seen) == [1, 2, 3]
    assert all(v > 60 for v in seen.values())
    one = corpus.ultrachat_cut(msgs[:2], random.Random(0))
    assert one == msgs[:1]
    assert corpus.ultrachat_cut([{"role": "assistant", "content": "x"}], random.Random(0)) == []


def test_code_bucket_and_counts():
    assert [corpus.code_bucket(x) for x in ("Python", "typescript", "javascript", "rust", "cpp", "")] == \
        ["python", "jsts", "jsts", "rust", "other", "other"]
    c = corpus.counts_by_share(40250, corpus.CODE_SHARE)
    assert c == {"python": 18113, "jsts": 8050, "rust": 6037, "other": 8050}


def test_summary():
    items = [{"id": 0, "kind": "chat", "source": "oasst", "split": "train", "prompt_ids": [1, 2, 3]},
             {"id": 1, "kind": "prose", "source": "wikitext", "split": "test", "prompt_ids": [1]}]
    s = corpus.summary(items)
    assert s["kind"] == {"chat": 1, "prose": 1} and s["split"] == {"train": 1, "test": 1}
    assert s["prompt_tokens"]["max"] == 3
