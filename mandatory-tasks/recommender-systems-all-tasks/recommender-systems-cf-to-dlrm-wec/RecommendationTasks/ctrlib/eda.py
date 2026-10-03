"""Exploratory statistics for the CTR data: balance, missingness, cardinality, OOV, interaction statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CAT_COLS, NUM_COLS


def _mi(codes_x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in mutual information (nats) between a discrete code array and binary y, Miller-Madow corrected."""
    n = len(y)
    joint = pd.crosstab(codes_x, y).to_numpy().astype(float)
    pxy = joint / n
    px = pxy.sum(1, keepdims=True); py = pxy.sum(0, keepdims=True)
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())
    k_cells = int(nz.sum()); kx = joint.shape[0]; ky = joint.shape[1]
    return max(0.0, mi - (k_cells - kx - ky + 1) / (2 * n))


def interaction_information(cat: np.ndarray, y: np.ndarray, max_card=60, cards=None, top=12):
    """II(i,j) = MI(y; x_i,x_j) - MI(y; x_i) - MI(y; x_j) for low-cardinality categorical fields (train split).
    Positive = the pair is jointly more informative than the sum of its parts (a synergy that explicit
    interaction layers, like DLRM's dot products, could exploit). Indicative only: positives are rare."""
    low = [k for k in range(cat.shape[1]) if cards[k] <= max_card]
    single = {k: _mi(cat[:, k], y) for k in low}
    rows = []
    for a in range(len(low)):
        for b in range(a + 1, len(low)):
            i, j = low[a], low[b]
            joint = cat[:, i].astype(np.int64) * (cards[j] + 1) + cat[:, j]
            rows.append({"pair": f"{CAT_COLS[i].split('_')[-1]}x{CAT_COLS[j].split('_')[-1]}",
                         "interaction_info": _mi(joint, y) - single[i] - single[j]})
    df = pd.DataFrame(rows).sort_values("interaction_info", ascending=False)
    return df.head(top).reset_index(drop=True), {CAT_COLS[k]: v for k, v in single.items()}


def compute_eda(d: dict) -> dict:
    raw, pre = d["raw"], d["pre"]
    tr, va, te = raw["train"], raw["val"], raw["test"]
    out = {
        "n_rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "positive_rate": {k: float(v.label.mean()) for k, v in raw.items()},
        "majority_class_accuracy_test": float(1 - te.label.mean()),
        "prior_log_loss_test": float(-(te.label.mean() * np.log(tr.label.mean())
                                       + (1 - te.label.mean()) * np.log(1 - tr.label.mean()))),
        "duplicate_rows_train": int(tr.duplicated().sum()),
    }
    miss = pd.concat([tr[NUM_COLS + CAT_COLS].isna().mean().rename("train"),
                      te[NUM_COLS + CAT_COLS].isna().mean().rename("test")], axis=1)
    out["missing_rate"] = miss
    spear = {c: float(tr[[c, "label"]].corr(method="spearman").iloc[0, 1]) for c in NUM_COLS}
    out["spearman_with_label"] = spear
    card_raw = {c: int(tr[c].fillna("__M__").nunique()) for c in CAT_COLS}
    out["cardinality_raw"] = card_raw
    out["cardinality_vocab"] = dict(zip(CAT_COLS, pre.cardinalities))
    out["oov_rate_val"] = dict(zip(CAT_COLS, (d["val"][1] == 0).mean(0).round(4)))
    out["oov_rate_test"] = dict(zip(CAT_COLS, (d["test"][1] == 0).mean(0).round(4)))
    out["total_embedding_rows"] = int(sum(pre.cardinalities))
    ii, single = interaction_information(d["train"][1], d["train"][2].astype(int), cards=pre.cardinalities)
    out["interaction_info"], out["single_mi"] = ii, single
    return out
