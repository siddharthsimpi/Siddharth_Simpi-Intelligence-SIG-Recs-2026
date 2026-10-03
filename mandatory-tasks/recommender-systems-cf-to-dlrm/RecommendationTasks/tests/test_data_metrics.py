import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ctrlib import data as D  # noqa: E402
from ctrlib import metrics as M  # noqa: E402


def _toy_frame(n=400, seed=0, cats=("a", "b", "c", "d")):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(-5, 100, n).astype(float) for c in D.NUM_COLS})
    df.loc[rng.random(n) < 0.3, D.NUM_COLS[2]] = np.nan
    for c in D.CAT_COLS:
        df[c] = rng.choice(list(cats), n)
    df["label"] = (rng.random(n) < 0.1).astype(int)
    return df


def test_metrics_match_hand_values_and_sklearn():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0]); p = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6, 0.5, 0.05])
    assert abs(M.roc_auc(y, p) - 13 / 15) < 1e-9                # 13 of 15 positive/negative pairs ordered correctly
    try:
        from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
    except ImportError:
        return
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.1).astype(int); p = np.clip(rng.random(500) * 0.5 + y * 0.1, 0.001, 0.999)
    assert abs(M.roc_auc(y, p) - roc_auc_score(y, p)) < 1e-9
    assert abs(M.average_precision(y, p) - average_precision_score(y, p)) < 1e-9
    assert abs(M.log_loss(y, p) - log_loss(y, p)) < 1e-9


def test_threshold_and_f1_selection():
    y = np.array([1, 1, 0, 0, 0, 0]); p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    thr = M.best_f1_threshold(y, p)
    m = M.threshold_metrics(y, p, thr)
    assert thr == 0.8 and m["f1"] == 1.0 and m["recall"] == 1.0
    mp, fo, cnt, ece = M.reliability(y, np.full(6, 1 / 3), n_bins=1)      # mean prediction == base rate -> ECE 0
    assert abs(ece) < 1e-9


def test_preprocessor_is_fit_on_train_only():
    tr, va = _toy_frame(400, 0), _toy_frame(200, 1, cats=("a", "b", "c", "d", "ZZZ"))
    pre = D.Preprocessor(min_count=3).fit(tr)
    dense_tr, cat_tr, _ = pre.transform(tr)
    dense_va, cat_va, _ = pre.transform(va)
    assert np.allclose(dense_tr[:, :13].mean(0), 0, atol=1e-5)            # standardised with TRAIN stats
    assert not np.allclose(dense_va[:, :13].mean(0), 0, atol=1e-3)        # validation is NOT re-standardised
    assert (cat_va[va[D.CAT_COLS[0]] == "ZZZ", 0] == 0).all()             # category unseen in train -> OOV id 0
    assert cat_tr.max() < max(pre.cardinalities) and cat_tr.min() >= 0
    assert pre.n_dense == 13 + len(pre.miss_cols) and np.isfinite(dense_va).all()


def test_rare_categories_go_to_oov():
    tr = _toy_frame(100, 0, cats=("a", "b"))
    tr.loc[0, D.CAT_COLS[0]] = "rare"
    pre = D.Preprocessor(min_count=3).fit(tr)
    assert "rare" not in pre.vocabs[0]
    assert pre.transform(tr)[1][0, 0] == 0


def test_stratified_split_is_disjoint_and_stratified():
    df = _toy_frame(1000, 3)
    df["row_id"] = np.arange(len(df))
    tr, va = D.stratified_split(df, seed=7, val_frac=0.2)
    assert not set(tr.row_id) & set(va.row_id) and len(tr) + len(va) == len(df)
    assert abs(va.label.mean() - df.label.mean()) < 0.01


def test_training_reduces_loss():
    from ctrlib.models import Schema
    from ctrlib.trainer import fit, predict_proba
    from ctrlib.models import build_model
    rng = np.random.default_rng(0)
    n = 3000
    cat = rng.integers(0, 5, (n, 3)); dense = rng.normal(size=(n, 2)).astype(np.float32)
    y = ((cat[:, 0] == 1) | (dense[:, 0] > 1.0)).astype(np.float32)           # learnable signal
    tr, va = (dense[:2400], cat[:2400], y[:2400]), (dense[2400:], cat[2400:], y[2400:])
    for spec in ({"type": "mlp", "emb_dim": 4, "hidden": [16]}, {"type": "dlrm", "emb_dim": 4, "bottom": [8], "top": [16]},
                 {"type": "dcn", "emb_dim": 4, "hidden": [16], "n_cross": 2}, {"type": "fm", "emb_dim": 4}):
        m = build_model(spec, Schema(2, [5, 5, 5]), 0)
        h = fit(m, tr, va, lr=1e-2, epochs=30, patience=30, seed=0)
        assert h["val_loss"][-1] < h["val_loss"][0] * 0.8, spec["type"]
        assert M.roc_auc(va[2], predict_proba(m, va)) > 0.9, spec["type"]


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("PASS", k)
