"""Loading, splitting and leakage-free preprocessing of the supplied CTR dataset (13 integer + 26 categorical).

Preprocessing is FITTED ON THE TRAINING PART ONLY and then applied to validation and test:
  numeric     : median imputation + missing-indicator, signed log1p (heavy tails), standardisation
  categorical : missing -> its own token; vocabulary = categories seen >= min_count times in train;
                everything else (rare in train, or never seen) -> shared id 0 ("OOV")
The original test file is never used for fitting, tuning, early stopping or threshold selection.
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
import pandas as pd

NUM_COLS = [f"integer_feature_{i}" for i in range(1, 14)]
CAT_COLS = [f"categorical_feature_{i}" for i in range(1, 27)]
ZIP_NAME = "dataset (tasks 2 and 3).zip"


def find_data(path: str | None = None):
    """Return (train_source, test_source). Accepts a dir with train.csv/test.csv, or the dataset zip."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))                 # repo root
    cands = [path] if path else []
    cands += [os.environ.get("CTR_DATA"), os.path.join(root, "datasets"), os.path.join(root, "datasets", ZIP_NAME),
              os.path.join(here, "..", "data"), "data", "datasets", os.path.join("datasets", ZIP_NAME)]
    for c in filter(None, cands):
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
            return os.path.join(c, "train.csv"), os.path.join(c, "test.csv")
        if os.path.isdir(c) and os.path.exists(os.path.join(c, ZIP_NAME)):
            c = os.path.join(c, ZIP_NAME)
        if os.path.isfile(c) and c.endswith(".zip"):
            return (c, "train.csv"), (c, "test.csv")
    raise FileNotFoundError("Could not find the CTR dataset. Pass --data <dir with train.csv/test.csv | dataset zip>.")


def _read(src) -> pd.DataFrame:
    if isinstance(src, tuple):                                             # (zip path, member)
        with zipfile.ZipFile(src[0]) as zf:
            name = next(n for n in zf.namelist() if n.endswith(src[1]))
            return pd.read_csv(io.BytesIO(zf.read(name)))
    return pd.read_csv(src)


def load_raw(path: str | None = None):
    tr_src, te_src = find_data(path)
    return _read(tr_src), _read(te_src)


def stratified_split(df: pd.DataFrame, seed: int = 42, val_frac: float = 0.2):
    """Stratified (by label) random train/validation split of the TRAIN file. Same split for every task/model."""
    rng = np.random.default_rng(seed)
    val_idx = []
    for lab in (0, 1):
        idx = np.where(df.label.values == lab)[0]
        val_idx.append(rng.choice(idx, size=int(round(val_frac * len(idx))), replace=False))
    val_idx = np.sort(np.concatenate(val_idx))
    mask = np.zeros(len(df), bool); mask[val_idx] = True
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


class Preprocessor:
    def __init__(self, min_count: int = 10, missing_indicators: bool = True):
        self.min_count, self.missing_indicators = min_count, missing_indicators

    @staticmethod
    def _slog(x):
        return np.sign(x) * np.log1p(np.abs(x))

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X = df[NUM_COLS].astype(float)
        self.medians = X.median().to_numpy()
        self.miss_cols = [k for k, c in enumerate(NUM_COLS) if X[c].isna().any()] if self.missing_indicators else []
        T = self._slog(X.fillna(pd.Series(self.medians, index=NUM_COLS)).to_numpy())
        self.mean, self.std = T.mean(0), T.std(0) + 1e-6
        self.vocabs = []
        for c in CAT_COLS:
            vc = df[c].fillna("__MISSING__").value_counts()
            keep = vc.index[vc.values >= self.min_count]
            self.vocabs.append({v: i + 1 for i, v in enumerate(keep)})      # 0 reserved for OOV
        self.cardinalities = [len(v) + 1 for v in self.vocabs]
        self.n_dense = len(NUM_COLS) + len(self.miss_cols)
        return self

    def transform(self, df: pd.DataFrame):
        X = df[NUM_COLS].astype(float)
        miss = X.isna().to_numpy()
        T = self._slog(X.fillna(pd.Series(self.medians, index=NUM_COLS)).to_numpy())
        dense = (T - self.mean) / self.std
        if self.miss_cols:
            dense = np.concatenate([dense, miss[:, self.miss_cols].astype(float)], axis=1)
        cat = np.empty((len(df), len(CAT_COLS)), dtype=np.int64)
        for k, c in enumerate(CAT_COLS):
            cat[:, k] = df[c].fillna("__MISSING__").map(self.vocabs[k]).fillna(0).astype(np.int64).to_numpy()
        return dense.astype(np.float32), cat, df.label.to_numpy().astype(np.float32)


def prepare(path=None, seed=42, val_frac=0.2, min_count=10):
    """Raw files -> (train, val, test) tuples of (dense, cat, y) + preprocessor + the raw frames."""
    raw_train, raw_test = load_raw(path)
    tr, va = stratified_split(raw_train, seed, val_frac)
    pre = Preprocessor(min_count).fit(tr)
    return {"train": pre.transform(tr), "val": pre.transform(va), "test": pre.transform(raw_test),
            "pre": pre, "raw": {"train": tr, "val": va, "test": raw_test}}
