"""Dataset loading and group-wise splitting."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def load_dataset(csv=None, image_dir=None, with_splits=True) -> pd.DataFrame:
    """Load train.csv (posting_id, image, image_phash, title, label_group).

    Adds a `path` column (absolute image path) and, if labels exist, a `split` column.
    Splits are done by label_group so that no product appears in two splits.
    """
    csv = Path(csv or C.DATA_DIR / "train.csv")
    df = pd.read_csv(csv)
    if C.MAX_GROUPS and "label_group" in df:
        rng = np.random.RandomState(C.SEED)
        groups = df["label_group"].unique()
        keep = rng.choice(groups, min(C.MAX_GROUPS, len(groups)), replace=False)
        df = df[df["label_group"].isin(keep)]
    df = df.reset_index(drop=True)
    image_dir = Path(image_dir or C.DATA_DIR / "train_images")
    df["path"] = [str(image_dir / f) for f in df["image"]]
    if with_splits and "label_group" in df:
        df["split"] = assign_splits(df)
    return df


def assign_splits(df: pd.DataFrame) -> np.ndarray:
    rng = np.random.RandomState(C.SEED)
    groups = df["label_group"].unique()
    rng.shuffle(groups)
    n = len(groups)
    n_tr, n_va = int(C.SPLIT_FRACS[0] * n), int(C.SPLIT_FRACS[1] * n)
    gmap = {}
    for k, g in enumerate(groups):
        gmap[g] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    return df["label_group"].map(gmap).values


def split_rows(df: pd.DataFrame) -> dict:
    """{'train': idx, 'val': idx, 'test': idx} with GLOBAL row indices into df."""
    return {s: np.where(df["split"].values == s)[0] for s in ("train", "val", "test")}


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    """Number of listings that share each row's label_group (including itself)."""
    return df["label_group"].map(df["label_group"].value_counts()).values


def get_tag(df: pd.DataFrame) -> str:
    return f"{len(df)}_{C.MAX_GROUPS}_{'smoke' if C.SMOKE else 'full'}"
