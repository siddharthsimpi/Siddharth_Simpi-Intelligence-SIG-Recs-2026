"""One-call evaluation of an embedding on the pair benchmark + the retrieval metric."""
import numpy as np

from . import config as C
from .metrics import (auc_ap, best_f1_threshold, best_retrieval_threshold, candidate_recall,
                      pair_metrics, retrieval_f1)
from .retrieval import pair_scores, topk_pairs


def pair_kind_auc(bench, s):
    """AUC of positives vs each negative type (reveals which negatives are hard)."""
    from sklearn.metrics import roc_auc_score
    out = {}
    pos = bench["kind"].values == "positive"
    for k in ("random_neg", "text_hard_neg", "image_hard_neg"):
        m = pos | (bench["kind"].values == k)
        if (bench["kind"].values == k).any():
            out[k] = float(roc_auc_score(bench["y"].values[m], s[m]))
    return out


def evaluate_embedding(E, df, rows, bench, gsize, metric="cos", k=C.TOPK):
    """Thresholds are chosen on VAL, every reported number is computed on TEST.

    E     : embedding aligned to df (dense or sparse). rows: {'val': ids, 'test': ids}
    bench : {'val': pairs_df, 'test': pairs_df}
    """
    lab, n = df["label_group"].values, len(df)
    res = {}
    # ---- pairwise
    bv, bt = bench["val"], bench["test"]
    sv = pair_scores(E, bv.i.values, bv.j.values, metric)
    st = pair_scores(E, bt.i.values, bt.j.values, metric)
    thr_p, f1_val = best_f1_threshold(bv.y.values, sv)
    pm = pair_metrics(bt.y.values, st, thr_p)
    auc, ap = auc_ap(bt.y.values, st)
    res.update(pair_thr=thr_p, pair_auc=auc, pair_ap=ap, pair_precision=pm["precision"],
               pair_recall=pm["recall"], pair_f1=pm["f1"], pair_f1_val=f1_val)
    res["auc_by_negative"] = pair_kind_auc(bt, st)
    # ---- retrieval (Kaggle-style)
    iv, jv, ssv = topk_pairs(E, rows["val"], k, metric)
    yv = (lab[iv] == lab[jv]).astype(int)
    thr_r, _, _ = best_retrieval_threshold(iv, jv, ssv, yv, rows["val"], gsize, n)
    it, jt, sst = topk_pairs(E, rows["test"], k, metric)
    yt = (lab[it] == lab[jt]).astype(int)
    r = retrieval_f1(it, jt, sst, yt, thr_r, rows["test"], gsize, n)
    res.update(retr_thr=thr_r, retr_f1=r["f1"], retr_precision=r["precision"], retr_recall=r["recall"],
               cand_recall=candidate_recall(it, yt, rows["test"], gsize, n), dim=int(E.shape[1]))
    return res
