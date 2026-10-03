"""Dataset loading, synthetic fallback, per-user splitting and matrix construction.

Primary dataset : MovieLens 100K (GroupLens) -- 943 users x 1682 movies, 100,000 explicit 1-5 star ratings.
Fallback        : a MovieLens-shaped synthetic dataset generated from a known latent-factor process
                  (used only when MovieLens cannot be downloaded, e.g. no internet).
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


@dataclass
class Dataset:
    name: str
    ratings: pd.DataFrame            # columns: user, item (contiguous 0-based indices), rating, timestamp
    n_users: int
    n_items: int
    item_titles: list[str] | None = None
    description: str = ""
    is_synthetic: bool = False
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- MovieLens 100K
def _parse_movielens(ml_dir: str) -> Dataset:
    raw = pd.read_csv(
        os.path.join(ml_dir, "u.data"), sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    # Re-index ids to contiguous 0..n-1 (id vocabulary only; no label information is used).
    user_ids = np.sort(raw.user_id.unique())
    item_ids = np.sort(raw.item_id.unique())
    umap = {v: k for k, v in enumerate(user_ids)}
    imap = {v: k for k, v in enumerate(item_ids)}
    df = pd.DataFrame({
        "user": raw.user_id.map(umap).astype(np.int64),
        "item": raw.item_id.map(imap).astype(np.int64),
        "rating": raw.rating.astype(np.float64),
        "timestamp": raw.timestamp.astype(np.int64),
    })
    titles = None
    item_path = os.path.join(ml_dir, "u.item")
    if os.path.exists(item_path):
        items = pd.read_csv(item_path, sep="|", header=None, encoding="latin-1", usecols=[0, 1])
        title_by_id = dict(zip(items[0], items[1]))
        titles = [title_by_id.get(i, f"item {i}") for i in item_ids]
    return Dataset(
        name="MovieLens-100K", ratings=df, n_users=len(user_ids), n_items=len(item_ids),
        item_titles=titles,
        description=("MovieLens 100K (GroupLens Research): 100,000 explicit 1-5 star ratings from 943 users on "
                     "1,682 movies; every user has rated at least 20 movies."),
    )


def _download(url: str) -> bytes:
    """Download with certificate verification; if the local CA store is broken (common on macOS python.org
    installs: CERTIFICATE_VERIFY_FAILED) retry with certifi's CA bundle, and finally without verification.
    The file is public data and its integrity is validated afterwards (100,000 rating lines)."""
    import ssl
    attempts = [("default certificates", None)]
    try:
        import certifi
        attempts.append(("certifi certificates", ssl.create_default_context(cafile=certifi.where())))
    except ImportError:
        pass
    attempts.append(("UNVERIFIED SSL (public dataset; integrity checked afterwards)", ssl._create_unverified_context()))
    last = None
    for label, ctx in attempts:
        try:
            with urllib.request.urlopen(url, timeout=60, context=ctx) as resp:
                if ctx is not None and "UNVERIFIED" in label:
                    print(f"[data] note: downloaded using {label}")
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"[data] download with {label} failed: {type(exc).__name__}")
    raise last


def load_movielens_100k(data_dir: str = "data", download: bool = True) -> Dataset:
    ml_dir = os.path.join(data_dir, "ml-100k")
    if not os.path.exists(os.path.join(ml_dir, "u.data")):
        if not download:
            raise FileNotFoundError(f"{ml_dir}/u.data not found and download disabled.")
        os.makedirs(data_dir, exist_ok=True)
        print(f"[data] downloading MovieLens-100K from {ML100K_URL} ...")
        payload = _download(ML100K_URL)
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            zf.extractall(data_dir)
        with open(os.path.join(ml_dir, "u.data"), "rb") as f:
            n_lines = sum(1 for _ in f)
        if n_lines != 100_000:
            raise ValueError(f"Downloaded u.data has {n_lines} lines, expected 100000 (corrupt download).")
    return _parse_movielens(ml_dir)


# --------------------------------------------------------------------------- synthetic fallback
def make_synthetic(seed: int = 42, n_users: int = 943, n_items: int = 1682,
                   n_ratings: int = 100_000, n_factors: int = 8) -> Dataset:
    """MovieLens-shaped synthetic ratings with a ground-truth low-rank structure.

    rating = clip(round(3.5 + b_u + b_i + p_u . q_i + noise), 1, 5)
    Long-tailed user activity and item popularity; popular/liked items are more likely to be rated.
    """
    rng = np.random.default_rng(seed)
    s = 0.5
    P = rng.normal(0, s, (n_users, n_factors))
    Q = rng.normal(0, s, (n_items, n_factors))
    bu = rng.normal(0, 0.3, n_users)
    bi = rng.normal(0, 0.5, n_items)
    affinity = P @ Q.T

    # long-tailed user activity (>= 20 ratings each, like MovieLens)
    extra = rng.lognormal(3.6, 0.9, n_users)
    extra = extra / extra.sum() * (n_ratings - 20 * n_users)
    counts = np.minimum(20 + np.floor(extra).astype(int), n_items // 2)

    pop = np.exp(rng.normal(0, 1.2, n_items) + 0.5 * bi)   # long-tailed item popularity
    users, items = [], []
    for u in range(n_users):
        w = pop * np.exp(0.5 * affinity[u])
        w /= w.sum()
        chosen = rng.choice(n_items, size=counts[u], replace=False, p=w)
        users.append(np.full(counts[u], u))
        items.append(chosen)
    users = np.concatenate(users)
    items = np.concatenate(items)
    raw = 3.5 + bu[users] + bi[items] + affinity[users, items] + rng.normal(0, 0.5, len(users))
    ratings = np.clip(np.rint(raw), 1, 5)
    df = pd.DataFrame({"user": users, "item": items, "rating": ratings,
                       "timestamp": np.arange(len(users))})
    return Dataset(
        name="Synthetic-MovieLens-like", ratings=df, n_users=n_users, n_items=n_items,
        description=("SYNTHETIC fallback dataset (MovieLens could not be downloaded): ratings generated from a "
                     "latent-factor model with user/item biases, long-tailed activity and popularity."),
        is_synthetic=True,
    )


def load_dataset(name: str = "auto", data_dir: str = "data", seed: int = 42) -> Dataset:
    if name == "synthetic":
        return make_synthetic(seed)
    if name == "movielens":
        return load_movielens_100k(data_dir)
    # auto
    try:
        return load_movielens_100k(data_dir)
    except Exception as exc:  # offline, blocked, corrupt zip, ...
        print("=" * 78)
        print(f"[data] WARNING: could not load MovieLens-100K ({type(exc).__name__}: {exc}).")
        print("[data] Falling back to a SYNTHETIC MovieLens-like dataset. Results are NOT real-world results.")
        print("[data] To use real data: download https://files.grouplens.org/datasets/movielens/ml-100k.zip in a browser\n[data] and unzip it into data/ so that data/ml-100k/u.data exists, then re-run.")
        print("=" * 78)
        return make_synthetic(seed)


# --------------------------------------------------------------------------- split / matrices
def per_user_split(df: pd.DataFrame, seed: int = 42, fractions=(0.8, 0.1, 0.1)):
    """Random per-user split into train/val/test (every user keeps >=1 val and >=1 test rating)."""
    assert abs(sum(fractions) - 1) < 1e-9
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, g in df.groupby("user", sort=True):
        idx = g.index.to_numpy()
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        n_val = max(1, int(round(fractions[1] * n)))
        n_test = max(1, int(round(fractions[2] * n)))
        parts["test"].append(idx[:n_test])
        parts["val"].append(idx[n_test:n_test + n_val])
        parts["train"].append(idx[n_test + n_val:])
    out = []
    for k in ("train", "val", "test"):
        out.append(df.loc[np.concatenate(parts[k])].sort_values(["user", "item"]).reset_index(drop=True))
    return tuple(out)


def to_matrices(df: pd.DataFrame, n_users: int, n_items: int):
    """Dense rating matrix R (0 where missing) and observation mask M (bool)."""
    R = np.zeros((n_users, n_items), dtype=np.float64)
    M = np.zeros((n_users, n_items), dtype=bool)
    R[df.user.values, df.item.values] = df.rating.values
    M[df.user.values, df.item.values] = True
    return R, M
