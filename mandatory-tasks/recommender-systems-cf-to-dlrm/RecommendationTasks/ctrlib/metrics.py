"""Binary CTR metrics implemented in NumPy: ROC-AUC, PR-AUC (average precision), log loss, threshold metrics, ECE."""
from __future__ import annotations

import numpy as np
import pandas as pd


def roc_auc(y, p) -> float:
    y = np.asarray(y); r = pd.Series(np.asarray(p)).rank(method="average").to_numpy()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, p) -> float:
    """Step-wise PR-AUC (same definition as sklearn.average_precision_score)."""
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    distinct = np.where(np.diff(ps))[0]
    idx = np.r_[distinct, len(ys) - 1]
    tp = np.cumsum(ys)[idx]; fp = (1 + idx) - tp
    precision = tp / (tp + fp); recall = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def log_loss(y, p, eps=1e-12) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def threshold_metrics(y, p, thr) -> dict:
    y = np.asarray(y); pred = np.asarray(p) >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": float(thr), "accuracy": float((pred == (y == 1)).mean()),
            "precision": prec, "recall": rec, "f1": f1}


def best_f1_threshold(y, p) -> float:
    """Threshold maximising F1, chosen on VALIDATION predictions only (then frozen for test)."""
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + y.sum())
    last = np.r_[np.where(np.diff(ps))[0], len(ys) - 1]          # only cut between distinct scores
    best = last[np.argmax(f1[last])]
    return float(ps[best])


def reliability(y, p, n_bins=10):
    """Quantile-binned reliability table: mean predicted prob, observed rate, count; plus ECE."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    order = np.argsort(p); bins = np.array_split(order, n_bins)
    mp = np.array([p[b].mean() for b in bins]); fo = np.array([y[b].mean() for b in bins])
    cnt = np.array([len(b) for b in bins])
    ece = float(np.sum(cnt * np.abs(mp - fo)) / cnt.sum())
    return mp, fo, cnt, ece


def roc_curve(y, p):
    y = np.asarray(y); order = np.argsort(-np.asarray(p), kind="mergesort")
    ys = y[order]; tpr = np.r_[0, np.cumsum(ys) / ys.sum()]; fpr = np.r_[0, np.cumsum(1 - ys) / (1 - ys).sum()]
    return fpr, tpr


def pr_curve(y, p):
    y = np.asarray(y); order = np.argsort(-np.asarray(p), kind="mergesort")
    ys = y[order]; tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
    return tp / ys.sum(), tp / k          # recall, precision


def all_metrics(y, p, thr) -> dict:
    mp, fo, cnt, ece = reliability(y, p)
    return {"roc_auc": roc_auc(y, p), "pr_auc": average_precision(y, p), "log_loss": log_loss(y, p),
            "ece": ece, **threshold_metrics(y, p, thr)}
