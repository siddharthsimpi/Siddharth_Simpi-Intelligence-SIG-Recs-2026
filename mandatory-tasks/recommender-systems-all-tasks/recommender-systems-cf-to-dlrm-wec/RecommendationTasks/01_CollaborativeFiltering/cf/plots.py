"""Plot helpers (matplotlib, headless)."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_eda(train_df, n_users, n_items, path, title=""):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    vc = train_df.rating.value_counts().sort_index()
    ax[0].bar(vc.index.astype(int), vc.values, color="#4C72B0")
    ax[0].set(title="Rating distribution (train)", xlabel="rating", ylabel="count")
    uc = np.sort(np.bincount(train_df.user, minlength=n_users))[::-1]
    ic = np.sort(np.bincount(train_df.item, minlength=n_items))[::-1]
    ax[1].loglog(np.arange(1, len(uc) + 1), np.maximum(uc, 1), label="users")
    ax[1].loglog(np.arange(1, len(ic) + 1), np.maximum(ic, 1), label="items")
    ax[1].set(title="Long tail: ratings per user / item", xlabel="rank", ylabel="# ratings")
    ax[1].legend()
    ax[2].hist(np.bincount(train_df.user, minlength=n_users), bins=40, color="#55A868")
    ax[2].set(title="Ratings per user (train)", xlabel="# ratings", ylabel="# users")
    fig.suptitle(title)
    _save(fig, path)


def plot_knn_tuning(tuning: dict, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (mode, shrink), pts in tuning.items():
        ks, rm = zip(*pts)
        ax.plot(ks, rm, marker="o", label=f"{mode}-{mode} (shrink={shrink:g})")
    ax.set(title="kNN CF: validation RMSE vs. neighbours k", xlabel="k", ylabel="validation RMSE")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_mf_curves(histories: dict, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ax[0].plot(h["epoch"], h["train_rmse"], label=name)
        ax[1].plot(h["epoch"], h["val_rmse"], label=name)
    ax[0].set(title="MF training RMSE", xlabel="epoch", ylabel="RMSE")
    ax[1].set(title="MF validation RMSE (early stopping)", xlabel="epoch", ylabel="RMSE")
    ax[0].legend(fontsize=8)
    _save(fig, path)


def plot_comparison(results, path):
    """results: list of dict(model, rmse, mae, ndcg@10, recall@10, ndcg_rated@5)."""
    names = [r["model"] for r in results]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    x = np.arange(len(names))
    w = 0.38
    ax[0].bar(x - w / 2, [r["rmse"] for r in results], w, label="RMSE")
    ax[0].bar(x + w / 2, [r["mae"] for r in results], w, label="MAE")
    ax[0].set(title="Rating prediction on test (lower = better)", xticks=x)
    ax[1].bar(x, [r["ndcg_rated@5"] for r in results], 0.6, color="#55A868")
    ax[1].set(title="Ordering of each user's rated test items: NDCG@5 (higher = better)", xticks=x)
    ax[1].set_ylim(min(r["ndcg_rated@5"] for r in results) * 0.95, 1.0)
    ax[2].bar(x - w / 2, [r["ndcg@10"] for r in results], w, label="NDCG@10")
    ax[2].bar(x + w / 2, [r["recall@10"] for r in results], w, label="Recall@10")
    ax[2].set(title="Full-catalogue top-10 (higher = better)", xticks=x)
    for a in ax:
        a.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
        a.title.set_fontsize(9)
    ax[0].legend()
    ax[2].legend()
    _save(fig, path)


def plot_buckets(user_tbl, item_tbl, path):
    """Each tbl: dict model -> (bucket_labels, rmse_values)."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a, tbl, ttl, xl in ((ax[0], user_tbl, "Test RMSE by user activity", "# train ratings of the user"),
                            (ax[1], item_tbl, "Test RMSE by item popularity", "# train ratings of the item")):
        for name, (labels, vals) in tbl.items():
            a.plot(range(len(labels)), vals, marker="o", label=name)
        labels = next(iter(tbl.values()))[0]
        a.set(title=ttl, xlabel=xl, ylabel="RMSE", xticks=range(len(labels)))
        a.set_xticklabels(labels)
        a.legend(fontsize=8)
    _save(fig, path)
