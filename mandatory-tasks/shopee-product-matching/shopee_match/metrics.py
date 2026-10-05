"""Metrics.

Two views of the same problem:
  * PAIR metrics     - "are these two listings the same product?" (AUC, AP, precision/recall/F1)
  * RETRIEVAL metric - the Kaggle competition metric: for every listing predict the SET of matching
                       listings (always including itself) and average the per-row F1 =
                       2*TP / (|predicted| + |true group|).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


# ---------------------------------------------------------------- pair metrics
def auc_ap(y, s):
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    b = int(np.argmax(f1[:-1]))
    return float(t[b]), float(f1[b])


def pair_metrics(y, s, thr):
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(thr=float(thr), precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-12),
                tp=tp, fp=fp, fn=fn, tn=tn)


def threshold_table(y, s, thrs):
    return pd.DataFrame([pair_metrics(y, s, t) for t in thrs])


# ---------------------------------------------------------------- retrieval metric
def retrieval_f1(i, j, s, y, thr, rows, gsize, n_total):
    """i, j, s, y: directed candidate pairs (global ids, score, same-group flag), self NOT included.
    rows: global ids being evaluated. gsize: group size per global id. Candidates below thr are ignored.
    Note: true matches that are not among the candidates count as misses (recall is bounded by retrieval)."""
    keep = s >= thr
    pc = np.bincount(i[keep], minlength=n_total)[rows]
    tpc = np.bincount(i[keep & (y == 1)], minlength=n_total)[rows]
    tp, npred, ntrue = 1 + tpc, 1 + pc, gsize[rows]
    return dict(f1=float(np.mean(2 * tp / (npred + ntrue))),
                precision=float(np.mean(tp / npred)), recall=float(np.mean(tp / ntrue)))


def best_retrieval_threshold(i, j, s, y, rows, gsize, n_total, n_grid=80):
    lo, hi = np.quantile(s, 0.2), s.max()
    best = (None, -1, None)
    for t in np.linspace(lo, hi, n_grid):
        r = retrieval_f1(i, j, s, y, t, rows, gsize, n_total)
        if r["f1"] > best[1]:
            best = (float(t), r["f1"], r)
    return best  # (thr, f1, dict)


def candidate_recall(i, y, rows, gsize, n_total):
    """Upper bound on recall: fraction of true matches that appear among the candidates at all."""
    tpc = np.bincount(i[y == 1], minlength=n_total)[rows]
    return float(np.mean(np.minimum((1 + tpc) / gsize[rows], 1.0)))
