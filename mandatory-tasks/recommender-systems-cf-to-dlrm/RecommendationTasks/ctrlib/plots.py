"""Plot helpers for the CTR tasks (matplotlib, headless)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import pr_curve, reliability, roc_curve


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_eda(eda, path):
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    pr = eda["positive_rate"]
    ax[0, 0].bar(list(pr), [100 * v for v in pr.values()], color="#4C72B0")
    ax[0, 0].set(title="Click rate by split (%) -- strong class imbalance", ylabel="% clicks")
    m = eda["missing_rate"]["train"]
    ax[0, 1].bar(range(len(m)), 100 * m.values, color=["#55A868"] * 13 + ["#C44E52"] * 26)
    ax[0, 1].set(title="Missing values in train (%): 13 numeric (green), 26 categorical (red)", xlabel="feature #")
    sp = eda["spearman_with_label"]
    ax[0, 2].bar(range(1, 14), list(sp.values()), color="#8172B2")
    ax[0, 2].set(title="Spearman corr(numeric feature, click)", xlabel="integer_feature #")
    ax[0, 2].axhline(0, color="k", lw=0.5)
    card = list(eda["cardinality_raw"].values())
    ax[1, 0].bar(range(1, 27), card, color="#CCB974"); ax[1, 0].set_yscale("log")
    ax[1, 0].set(title="Categorical cardinality in train (log scale)", xlabel="categorical_feature #")
    ii = eda["interaction_info"]
    ax[1, 1].barh(ii.pair[::-1], ii.interaction_info[::-1], color="#64B5CD")
    ax[1, 1].set(title="Top pairwise interaction information (low-card fields)", xlabel="nats (bias-corrected)")
    oov = list(eda["oov_rate_test"].values())
    ax[1, 2].bar(range(1, 27), 100 * np.array(oov), color="#C44E52")
    ax[1, 2].set(title="Test values mapped to OOV bucket (%)", xlabel="categorical_feature #")
    _save(fig, path)


def plot_curves_grid(runs, path, ncols=4):
    n = len(runs); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    for ax, r in zip(axes.ravel(), runs):
        h = r["history"]
        ax.plot(h["epoch"], h["train_loss"], label="train"); ax.plot(h["epoch"], h["val_loss"], label="val")
        ax.axvline(h["best_epoch"], color="gray", ls=":", lw=1)
        ax.set_title(r["name"], fontsize=7); ax.set_xlabel("epoch"); ax.set_ylabel("log loss")
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    _save(fig, path)


def plot_reliability(preds: dict, path, n_bins=10):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (y, p) in preds.items():
        mp, fo, cnt, ece = reliability(y, p, n_bins)
        ax[0].plot(mp, fo, marker="o", label=f"{name} (ECE {ece:.4f})")
    lim = max(ax[0].get_xlim()[1], ax[0].get_ylim()[1])
    ax[0].plot([0, lim], [0, lim], "k--", lw=1)
    ax[0].set(title="Reliability diagram (test, quantile bins)", xlabel="mean predicted CTR", ylabel="observed CTR")
    ax[0].legend(fontsize=7)
    for name, (y, p) in preds.items():
        ax[1].hist(p, bins=60, histtype="step", label=name, range=(0, 0.3))
    ax[1].set(title="Predicted CTR distribution (test)", xlabel="predicted probability", yscale="log")
    _save(fig, path)


def plot_roc_pr(preds: dict, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, (y, p) in preds.items():
        f, t = roc_curve(y, p); ax[0].plot(f, t, label=name)
        r, pr = pr_curve(y, p); ax[1].plot(r, pr, label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR"); ax[0].legend(fontsize=7)
    base = float(np.mean(next(iter(preds.values()))[0]))
    ax[1].axhline(base, color="k", ls="--", lw=1, label=f"no-skill {base:.3f}")
    ax[1].set(title="Precision-recall (test)", xlabel="recall", ylabel="precision", ylim=(0, 0.3)); ax[1].legend(fontsize=7)
    _save(fig, path)


def plot_bars(df, path, title="Test metrics (mean ± std over seeds)"):
    names = list(df.model)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for a, (m, lab, better) in zip(ax, [("roc_auc", "ROC-AUC", "higher"), ("pr_auc", "PR-AUC", "higher"),
                                        ("log_loss", "log loss", "lower")]):
        a.bar(range(len(df)), df[m], yerr=df[m + "_std"], capsize=3, color="#4C72B0")
        a.set(title=f"{lab} ({better} = better)", xticks=range(len(df)))
        a.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
        lo = float((df[m] - df[m + "_std"]).min()); hi = float((df[m] + df[m + "_std"]).max())
        a.set_ylim(lo - 0.15 * (hi - lo) - 1e-4, hi + 0.15 * (hi - lo) + 1e-4)
    fig.suptitle(title)
    _save(fig, path)


def plot_cost(df, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    x = np.arange(len(df))
    ax[0].bar(x, df.embedding_params / 1e3, label="embedding", color="#C44E52")
    ax[0].bar(x, df.dense_params / 1e3, bottom=df.embedding_params / 1e3, label="dense/MLP", color="#4C72B0")
    ax[0].set(title="Parameters (thousands)", xticks=x); ax[0].legend(fontsize=8)
    ax[1].bar(x, df.sec_per_epoch, color="#55A868"); ax[1].set(title="Training seconds per epoch (1 CPU, NumPy)", xticks=x)
    ax[2].bar(x, df.infer_us_per_row, color="#8172B2"); ax[2].set(title="Inference µs per row", xticks=x)
    for a in ax:
        a.set_xticklabels(df.model, rotation=25, ha="right", fontsize=7)
    _save(fig, path)


def plot_ablation(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, m, lab in zip(ax, ["roc_auc", "log_loss"], ["ROC-AUC (test)", "log loss (test)"]):
        a.barh(range(len(df)), df[m], xerr=df[m + "_std"], capsize=3, color="#64B5CD")
        a.set(title=lab, yticks=range(len(df))); a.set_yticklabels(df.model, fontsize=7); a.invert_yaxis()
        lo = float((df[m] - df[m + "_std"]).min()); hi = float((df[m] + df[m + "_std"]).max())
        a.set_xlim(lo - 0.1 * (hi - lo) - 1e-4, hi + 0.1 * (hi - lo) + 1e-4)
    _save(fig, path)
