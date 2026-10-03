"""Rating-prediction metrics (RMSE/MAE) and top-K ranking metrics."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def ranking_metrics(scores: np.ndarray, exclude_mask: np.ndarray, relevant_mask: np.ndarray,
                    ks=(5, 10, 20), seed: int = 0) -> dict:
    """Top-K ranking metrics averaged over users that have >=1 relevant held-out item.

    scores        (U, I) predicted preference, higher = better
    exclude_mask  (U, I) True for items that must not be recommended (already seen in train/val)
    relevant_mask (U, I) True for held-out items counted as relevant (e.g. test rating >= 4)
    Ties are broken randomly (tiny noise) so that constant scores behave like a random ranking.
    """
    rng = np.random.default_rng(seed)
    users = np.where(relevant_mask.sum(1) > 0)[0]
    s = scores[users].astype(np.float64) + rng.random((len(users), scores.shape[1])) * 1e-7
    s[exclude_mask[users]] = -np.inf
    kmax = max(ks)
    top = np.argpartition(-s, kmax - 1, axis=1)[:, :kmax]
    order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
    top = np.take_along_axis(top, order, axis=1)                          # (n, kmax) ranked item ids
    rel = relevant_mask[users]
    hits = np.take_along_axis(rel, top, axis=1).astype(np.float64)       # (n, kmax)
    n_rel = rel.sum(1).astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, kmax + 2))

    out = {"n_eval_users": int(len(users))}
    for k in ks:
        h = hits[:, :k]
        dcg = (h * discounts[:k]).sum(1)
        ideal_len = np.minimum(n_rel, k).astype(int)
        idcg = np.array([discounts[:m].sum() for m in ideal_len])
        out[f"precision@{k}"] = float(np.mean(h.sum(1) / k))
        out[f"recall@{k}"] = float(np.mean(h.sum(1) / n_rel))
        out[f"ndcg@{k}"] = float(np.mean(dcg / idcg))
        out[f"hitrate@{k}"] = float(np.mean(h.sum(1) > 0))
        out[f"coverage@{k}"] = float(len(np.unique(top[:, :k])) / scores.shape[1])
    return out


def rated_ndcg(scores: np.ndarray, test_mask: np.ndarray, test_ratings: np.ndarray, k: int = 5, seed: int = 0) -> float:
    """Graded NDCG@k when each user's candidates are only his/her held-out rated items (gain 2^r - 1).

    Measures whether a model *orders* items the user is known to have rated according to how much they liked
    them -- the question explicit-rating CF is actually trained for. It ignores which items users chose to rate
    (exposure), unlike the full-catalogue ranking in `ranking_metrics`.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for u in np.where(test_mask.sum(1) >= 2)[0]:
        idx = np.where(test_mask[u])[0]
        s = scores[u, idx] + rng.random(len(idx)) * 1e-7
        gain = 2.0 ** test_ratings[u, idx] - 1.0
        disc = 1.0 / np.log2(np.arange(2, min(k, len(idx)) + 2))
        dcg = (gain[np.argsort(-s)[:k]] * disc).sum()
        idcg = (np.sort(gain)[::-1][:k] * disc).sum()
        vals.append(dcg / idcg)
    return float(np.mean(vals))
