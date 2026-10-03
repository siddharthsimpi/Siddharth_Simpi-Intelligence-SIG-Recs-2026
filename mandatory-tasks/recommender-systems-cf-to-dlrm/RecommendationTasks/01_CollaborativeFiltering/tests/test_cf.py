import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from cf.data import make_synthetic, per_user_split, to_matrices
from cf.memory_cf import KNNCF, weighted_average_prediction
from cf.metrics import ranking_metrics, rmse
from cf.mf import BiasedMF


def close(a, b, abs_tol=1e-9):
    return math.isclose(a, b, rel_tol=0, abs_tol=abs_tol)


def test_readme_worked_example():
    # Task README: Bob (0.9) rated Avatar 4, Dave (0.8) rated 5 -> ~4.47
    assert close(weighted_average_prediction([0.9, 0.8], [4, 5]), 4.4706, 1e-3)


def test_split_is_disjoint_and_complete():
    ds = make_synthetic(0, n_users=60, n_items=120, n_ratings=3000)
    tr, va, te = per_user_split(ds.ratings, 0)
    keys = lambda d: set(zip(d.user, d.item))
    assert not (keys(tr) & keys(va)) and not (keys(tr) & keys(te)) and not (keys(va) & keys(te))
    assert len(tr) + len(va) + len(te) == len(ds.ratings)
    assert set(va.user) == set(te.user) == set(ds.ratings.user)


def test_mf_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    m = BiasedMF(6, 7, n_factors=3, reg=0.1, seed=1, init_std=0.5)
    m.mu = 3.0
    for k in ("bu", "bi"):
        m.params[k] = rng.normal(0, 0.3, m.params[k].shape)
    u, i, r = rng.integers(0, 6, 20), rng.integers(0, 7, 20), rng.integers(1, 6, 20).astype(float)
    _, g = m.loss_and_grads(u, i, r)
    eps = 1e-6
    for name, arr in m.params.items():
        for idx in np.ndindex(*arr.shape):
            old = arr[idx]
            arr[idx] = old + eps; lp, _ = m.loss_and_grads(u, i, r)
            arr[idx] = old - eps; lm, _ = m.loss_and_grads(u, i, r)
            arr[idx] = old
            assert close(g[name][idx], (lp - lm) / (2 * eps), 1e-6), (name, idx)


def test_mf_recovers_low_rank_structure_and_beats_global_mean():
    ds = make_synthetic(1, n_users=120, n_items=150, n_ratings=8000)
    tr, va, te = per_user_split(ds.ratings, 1)
    m = BiasedMF(ds.n_users, ds.n_items, 8, lr=0.01, reg=0.05, epochs=40, seed=1)
    m.fit(tr.user, tr.item, tr.rating, val=(va.user.values, va.item.values, va.rating.values))
    base = rmse(te.rating, np.full(len(te), tr.rating.mean()))
    assert rmse(te.rating, m.predict_pairs(te.user.values, te.item.values)) < base
    assert m.history["train_rmse"][-1] < m.history["train_rmse"][0]


def test_knn_predictions_in_range_and_better_than_mean():
    for mode in ("user", "item"):
        _check_knn(mode)


def _check_knn(mode):
    ds = make_synthetic(2, n_users=120, n_items=150, n_ratings=8000)
    tr, va, te = per_user_split(ds.ratings, 2)
    R, M = to_matrices(tr, ds.n_users, ds.n_items)
    p = KNNCF(mode, k=30, shrink=25).fit(R, M).predict_all()
    assert p.shape == R.shape and p.min() >= 1 and p.max() <= 5 and np.isfinite(p).all()
    pt = p[te.user.values, te.item.values]
    assert rmse(te.rating, pt) < rmse(te.rating, np.full(len(te), tr.rating.mean()))


def test_ranking_metrics_perfect_and_worst():
    scores = np.array([[0.9, 0.8, 0.1, 0.0], [0.0, 0.1, 0.8, 0.9]])
    rel = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], bool)
    exc = np.zeros_like(rel)
    perfect = ranking_metrics(scores, exc, rel, ks=(2,))
    assert close(perfect["precision@2"], 1) and close(perfect["ndcg@2"], 1) and close(perfect["recall@2"], 1)
    worst = ranking_metrics(-scores, exc, rel, ks=(2,))
    assert worst["precision@2"] == 0 and worst["ndcg@2"] == 0


def test_ranking_excludes_seen_items():
    scores = np.array([[1.0, 0.9, 0.1]])
    rel = np.array([[0, 0, 1]], bool)
    exc = np.array([[1, 1, 0]], bool)       # top-scored items already seen
    out = ranking_metrics(scores, exc, rel, ks=(1,))
    assert out["hitrate@1"] == 1.0


if __name__ == "__main__":      # lets you run the tests without pytest:  python tests/test_cf.py
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
