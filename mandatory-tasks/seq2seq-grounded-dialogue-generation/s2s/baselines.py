"""Non-neural baselines for dialogue: TF-IDF retrieval of a training reply, and extractive document-sentence selection."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np


class TfidfIndex:
    """Sparse TF-IDF cosine retrieval with an inverted index (pure Python / NumPy)."""

    def __init__(self, docs):
        n = len(docs)
        df = Counter(t for d in docs for t in set(d))
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
        self.post = defaultdict(list)
        self.norms = np.zeros(n)
        for i, d in enumerate(docs):
            vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in Counter(d).items()}
            self.norms[i] = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            for t, v in vec.items():
                self.post[t].append((i, v))
        self.n = n

    def query(self, tokens, k=1):
        q = {t: (1 + math.log(c)) * self.idf[t] for t, c in Counter(tokens).items() if t in self.idf}
        if not q:
            return []
        scores = np.zeros(self.n)
        for t, qv in q.items():
            for i, v in self.post[t]:
                scores[i] += qv * v
        scores /= self.norms
        top = np.argsort(-scores)[:k]
        return [int(i) for i in top if scores[i] > 0]


def retrieval_baseline(train_examples, test_examples, last_turn_only=True):
    """Reply of the training example whose history is most similar to the test history (ungrounded)."""
    def key(e):
        h = e["hist_tok"]
        if last_turn_only:
            cut = max((i for i, t in enumerate(h) if t in ("<usr1>", "<usr2>")), default=0)
            h = h[cut:]
        return [t for t in h if not t.startswith("<")]
    idx = TfidfIndex([key(e) for e in train_examples])
    out = []
    for e in test_examples:
        r = idx.query(key(e), 1)
        out.append(train_examples[r[0]]["resp_tok"] if r else [])
    return out


def split_sentences(tokens):
    sents, cur = [], []
    for t in tokens:
        cur.append(t)
        if t in (".", "!", "?", "\u0964"):
            sents.append(cur); cur = []
    if cur:
        sents.append(cur)
    return sents


def extractive_baseline(test_examples, train_examples, max_len=25):
    """Grounded but non-generative: return the document sentence most similar to the dialogue history."""
    idf_index = TfidfIndex([e["doc_tok"] for e in train_examples])
    out = []
    for e in test_examples:
        sents = split_sentences(e["doc_tok"]) or [[]]
        q = Counter(t for t in e["hist_tok"] if not t.startswith("<"))
        def score(s):
            c = Counter(s)
            return sum(q[t] * idf_index.idf.get(t, 1.0) for t in c if t in q) / math.sqrt(len(s) + 1)
        best = max(sents, key=score)
        out.append(best[:max_len])
    return out
