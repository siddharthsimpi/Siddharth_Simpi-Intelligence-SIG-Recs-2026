"""Exact (brute-force, chunked) nearest-neighbour search and pair scoring."""
import numpy as np
import scipy.sparse as sp


def topk_search(E, k, metric="cos", chunk=1024):
    """For every row find its k most similar OTHER rows.

    metric: 'cos' (inputs should be L2-normalised), 'dot', or 'l2' (score = -distance).
    Returns (idx, score), both (n, k), sorted by descending score, self excluded.
    Memory is O(chunk * n); time is O(n^2 * d) - fine up to ~40k rows, beyond that use ANN (faiss).
    """
    n = E.shape[0]
    k = min(k, n - 1)
    sparse = sp.issparse(E)
    ET = E.T.tocsr() if sparse else E.T
    sq = None if sparse or metric != "l2" else (E ** 2).sum(1)
    idx_all, sc_all = np.zeros((n, k), np.int64), np.zeros((n, k), np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        if sparse:
            S = (E[a:b] @ ET).toarray()
        elif metric == "l2":
            S = -np.sqrt(np.maximum(sq[a:b, None] + sq[None, :] - 2 * E[a:b] @ ET, 0))
        else:
            S = E[a:b] @ ET
        S[np.arange(b - a), np.arange(a, b)] = -np.inf      # exclude self
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sc = np.take_along_axis(S, part, 1)
        order = np.argsort(-sc, axis=1)
        idx_all[a:b] = np.take_along_axis(part, order, 1)
        sc_all[a:b] = np.take_along_axis(sc, order, 1)
    return idx_all, sc_all


def topk_pairs(E, rows, k, metric="cos", chunk=1024):
    """Top-k search restricted to `rows` (global indices). Returns directed pairs (i, j, score) as GLOBAL ids."""
    rows = np.asarray(rows)
    idx, sc = topk_search(E[rows], k, metric, chunk)
    return np.repeat(rows, idx.shape[1]), rows[idx].ravel(), sc.ravel()


def pair_scores(E, i, j, metric="cos", chunk=None):
    """Score given pairs (global ids) without building any n x n matrix.
    Processed in chunks so that E[i], E[j] never materialise millions of rows at once."""
    i, j = np.asarray(i), np.asarray(j)
    sparse = sp.issparse(E)
    chunk = chunk or (100_000 if sparse else max(5_000, int(2e8 // max(E.shape[1], 1))))
    out = np.empty(len(i), np.float32)
    for a in range(0, len(i), chunk):
        ia, ja = i[a:a + chunk], j[a:a + chunk]
        if sparse:
            out[a:a + chunk] = np.asarray(E[ia].multiply(E[ja]).sum(1)).ravel()
        elif metric == "l2":
            out[a:a + chunk] = -np.linalg.norm(E[ia] - E[ja], axis=1)
        else:
            out[a:a + chunk] = np.einsum("ij,ij->i", E[ia], E[ja])
    return out
