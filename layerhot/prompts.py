"""Deterministic prompt mix (chat, code, prose) with a fixed train/test split.

Chat and code prompts go through the chat template with thinking disabled.
Prose prompts are raw wikitext-103 paragraph openings that the model continues.
The split is by prompt, so no test prompt shares text with a training prompt.
"""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path

WIKITEXT = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / \
    "datasets--Salesforce--wikitext"

CHAT_TOPICS = [
    "how vaccines train the immune system", "why the sky is blue", "how a bill becomes law in the US",
    "the causes of the French Revolution", "how compound interest works", "what inflation is",
    "how to prepare for a job interview", "how photosynthesis works", "the difference between weather and climate",
    "how to start running as a beginner", "what a black hole is", "how credit scores work",
    "how to write a good cover letter", "the history of the printing press", "how noise-cancelling headphones work",
    "why we dream", "how to keep houseplants alive", "how GPS works", "what the stock market is",
    "how to learn a new language as an adult", "the water cycle", "how airplanes stay in the air",
    "how to make sourdough bread", "the causes of World War I", "what DNA is and what it does",
    "how electric cars work", "how to manage stress at work", "the rules of chess for a beginner",
    "how the internet routes a message", "what a healthy breakfast looks like", "how tides work",
    "how to negotiate a salary", "the main ideas of stoicism", "how earthquakes happen",
    "how to give constructive feedback", "why bees matter for agriculture", "how a refrigerator keeps food cold",
    "how to plan a week of meals on a budget", "the life cycle of a star", "how antibiotics work",
    "how to write a clear email to a manager", "what machine learning is", "how volcanoes form",
    "how to read a nutrition label", "the history of the Olympic Games", "how solar panels make electricity",
    "how to choose a first programming language", "why the Roman Empire fell", "how coffee is grown and processed",
    "how to improve sleep quality",
]
CHAT_FORMS = [
    "Explain {t} in simple terms.",
    "Write a short guide on {t}.",
    "I'm a high school student. Can you tell me about {t}?",
]

CODE_TASKS = [
    "reverse a singly linked list", "check whether a string is a palindrome", "compute the nth Fibonacci number iteratively",
    "merge two sorted arrays", "find the most frequent word in a text", "implement binary search",
    "parse a CSV line that may contain quoted commas", "implement an LRU cache", "flatten a nested list",
    "validate an email address with a regular expression", "compute the edit distance between two strings",
    "implement a stack with push, pop and min in O(1)", "count the islands in a 2D grid", "implement quicksort",
    "read a JSON file and print the keys at the top level", "group anagrams from a list of words",
    "implement a simple rate limiter", "detect a cycle in a directed graph", "convert a Roman numeral to an integer",
    "find the longest common prefix of a list of strings", "implement a trie with insert and search",
    "serialize and deserialize a binary tree", "compute the median of a stream of numbers",
    "implement a thread-safe counter", "write a function that retries an HTTP request with exponential backoff",
    "implement Dijkstra's shortest path algorithm", "rotate a matrix by 90 degrees", "tokenize an arithmetic expression",
    "implement a min-heap", "deduplicate rows in a list of dictionaries by a key",
    "write a command line tool that counts lines in files", "implement matrix multiplication",
    "validate a Sudoku board", "find all permutations of a string", "implement a debounce function",
]
CODE_LANGS = ["Python", "JavaScript", "C++", "Rust", "Go"]
CODE_FORMS = [
    "Write a {lang} function to {t}. Include a short explanation.",
    "In {lang}, how do I {t}? Show the code.",
]


def _detok(s: str) -> str:
    import re

    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([,.;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[$]) ", r"\1", s)
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s)
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    return s


def _wikitext_paragraphs(n: int, seed: int) -> list[str]:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(str(WIKITEXT / "snapshots" / "*" / "wikitext-103-raw-v1" / "train-00000-*.parquet")))
    if not files:
        raise FileNotFoundError("wikitext-103-raw-v1 not found in HF cache")
    texts = pq.read_table(files[0], columns=["text"]).column("text").to_pylist()
    paras = [_detok(t.strip()) for t in texts
             if len(t) > 600 and not t.strip().startswith("=")]
    rng = random.Random(seed)
    rng.shuffle(paras)
    out = []
    for p in paras[:n]:
        words = p.split()
        out.append(" ".join(words[:60]))
    return out


def build(seed: int = 0, n_prose: int = 150) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for ti, t in enumerate(CHAT_TOPICS):
        for f in CHAT_FORMS:
            items.append({"kind": "chat", "text": f.format(t=t), "group": f"chat:{ti}"})
    for ti, t in enumerate(CODE_TASKS):
        for lang in CODE_LANGS:
            f = CODE_FORMS[rng.randrange(len(CODE_FORMS))]
            items.append({"kind": "code", "text": f.format(lang=lang, t=t), "group": f"code:{ti}"})
    for pi, p in enumerate(_wikitext_paragraphs(n_prose, seed)):
        items.append({"kind": "prose", "text": p, "group": f"prose:{pi}"})
    rng.shuffle(items)
    # split by group (topic / task / paragraph): no test prompt shares a topic with training
    groups = sorted({it["group"] for it in items})
    random.Random(seed + 1).shuffle(groups)
    test_groups = set(groups[: len(groups) // 5])
    for i, it in enumerate(items):
        it["id"] = i
        it["split"] = "test" if it["group"] in test_groups else "train"
    return items


def encode(tokenizer, item: dict) -> list[int]:
    if item["kind"] == "prose":
        return tokenizer.encode(item["text"])
    msgs = [{"role": "user", "content": item["text"]}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


if __name__ == "__main__":
    items = build()
    from collections import Counter
    print(len(items), Counter(i["kind"] for i in items), Counter(i["split"] for i in items))
    for it in items[:4]:
        print(it)
