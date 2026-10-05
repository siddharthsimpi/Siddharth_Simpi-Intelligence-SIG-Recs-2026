"""Benchmark pair sets (fixed per split) with random, text-hard and image-hard negatives."""
from itertools import combinations

import numpy as np
import pandas as pd

import pickle

from . import config as C
from .retrieval import topk_search
from .text import normalize_title, tfidf_matrix

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash_u64(series) -> np.ndarray:
    return np.array([int(str(h), 16) for h in series], dtype=np.uint64)


def hamming(a, b):
    """Hamming distance between 64-bit hashes; a (c,) vs b (n,) -> (c, n)."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return _POP[x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)].sum(-1)


def pair_hamming(ph, i, j):
    """Element-wise Hamming distance for pair arrays."""
    x = np.bitwise_xor(ph[i], ph[j])
    return _POP[x.view(np.uint8).reshape(-1, 8)].sum(1)


def hamming_topk(ph, k, chunk=256, seed=0):
    rng = np.random.RandomState(seed)
    n = len(ph)
    k = min(k, n - 1)
    out = np.zeros((n, k), np.int64)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        D = hamming(ph[a:b], ph).astype(np.float32) + rng.rand(b - a, n).astype(np.float32) * 0.5
        D[np.arange(b - a), np.arange(a, b)] = 1e9
        out[a:b] = np.argpartition(D, k - 1, axis=1)[:, :k]
    return out


def build_benchmark_pairs(df, rows, seed=42, max_pos=20000, k_hard=10) -> pd.DataFrame:
    """Balanced pair set for ONE split. Columns: i, j (GLOBAL ids), y, kind.

    positives : same label_group (sampled to max_pos)
    negatives : 1/3 random, 1/3 'text-hard' (nearest by char TF-IDF but different product),
                1/3 'image-hard' (nearest by perceptual-hash Hamming distance but different product)
    Using cheap, model-independent miners keeps the benchmark identical for Parts B, C and Finale.
    """
    rng = np.random.RandomState(seed)
    rows = np.asarray(rows)
    sub = df.iloc[rows]
    lab, n = sub["label_group"].values, len(sub)

    pos = []
    for _, idxs in pd.Series(np.arange(n)).groupby(lab):
        if len(idxs) > 1:
            pos.extend(combinations(idxs.values, 2))
    pos = np.array(pos).reshape(-1, 2)
    if len(pos) > max_pos:
        pos = pos[rng.choice(len(pos), max_pos, replace=False)]
    N = len(pos)
    n_each = max(N // 3, 1)

    def sample(cand, m):
        return cand[rng.choice(len(cand), min(m, len(cand)), replace=False)] if len(cand) else cand.reshape(0, 2)

    a, b = rng.randint(0, n, n_each * 4), rng.randint(0, n, n_each * 4)
    ok = lab[a] != lab[b]
    rnd = sample(np.c_[a[ok], b[ok]], n_each)

    X = tfidf_matrix([normalize_title(t) for t in sub["title"]], "char_wb", (2, 5), min_df=1)
    idx, _ = topk_search(X, k_hard)
    A, B = np.repeat(np.arange(n), idx.shape[1]), idx.ravel()
    m = lab[A] != lab[B]
    txt = sample(np.c_[A[m], B[m]], n_each)

    hidx = hamming_topk(phash_u64(sub["image_phash"]), k_hard, seed=seed)
    A, B = np.repeat(np.arange(n), hidx.shape[1]), hidx.ravel()
    m = lab[A] != lab[B]
    img = sample(np.c_[A[m], B[m]], n_each)

    parts = [(pos, 1, "positive"), (rnd, 0, "random_neg"), (txt, 0, "text_hard_neg"), (img, 0, "image_hard_neg")]
    out = pd.concat([pd.DataFrame({"i": rows[p[:, 0]], "j": rows[p[:, 1]], "y": y, "kind": k})
                     for p, y, k in parts if len(p)], ignore_index=True)
    key = np.sort(out[["i", "j"]].values, axis=1)
    out = out.loc[~pd.Series(map(tuple, key)).duplicated().values].reset_index(drop=True)
    return out


def get_benchmark(df, rows, tag, splits=("val", "test"), **kw):
    """Cached {split: pair_df}. Same pairs for every part => directly comparable numbers."""
    out = {}
    for s in splits:
        f = C.CACHE_DIR / tag / f"bench_{s}.pkl"
        if f.exists():
            out[s] = pickle.load(open(f, "rb"))
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            out[s] = build_benchmark_pairs(df, rows[s], seed=C.SEED, **kw)
            pickle.dump(out[s], open(f, "wb"))
    return out
