"""Inference with the saved final model on ANY csv that has posting_id, image, image_phash, title.

    python Finale/src/predict.py --csv data/shopee-product-matching/train.csv \
        --images data/shopee-product-matching/train_images --out predictions.csv

Output: posting_id,matches  (space-separated posting_ids, always including itself) - Kaggle submission format.
Note: TF-IDF vocabularies/IDF are re-fitted on the file you pass (unsupervised), as during training.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib

import joblib
import numpy as np
import pandas as pd

from shopee_match import config as C
from shopee_match.data import load_dataset
from shopee_match.embeddings import get_embedding
from src.features import PairFeaturizer, candidate_pairs
from src.fusion import gbm_score, weighted_score
from src.pipeline import build_match_lists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--model", default=str(C.ROOT / "Finale" / "results" / "final_model.joblib"))
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()

    bundle = joblib.load(a.model)
    df = load_dataset(a.csv, a.images, with_splits=False)
    tag = "pred_" + hashlib.md5(f"{a.csv}{len(df)}".encode()).hexdigest()[:8]
    labels = df["label_group"].values if "label_group" in df else np.arange(len(df))
    mods = list(dict.fromkeys(bundle["text_mods"] + bundle["img_mods"]))
    E = {m: get_embedding(df, m, tag) for m in mods}
    rows = np.arange(len(df))
    cand = candidate_pairs(E, bundle["cand_mods"], rows, bundle["cand_k"], labels)
    F = PairFeaturizer(df, E, bundle["text_mods"], bundle["img_mods"])(cand)
    if bundle["kind"] == "gbm":
        score = gbm_score(bundle["model"], F, bundle["cols"])
    else:
        score = weighted_score(F, bundle["text_cols"], bundle["img_cols"], bundle["w_text"])
    keep = score >= bundle["thr"]
    m = build_match_lists(rows, cand.i.values[keep], cand.j.values[keep], bundle.get("post", "none"), len(df))
    pid = df["posting_id"].values
    out = pd.DataFrame({"posting_id": pid, "matches": [" ".join(pid[x] for x in m[int(i)]) for i in rows]})
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} rows, mean #matches/row = {np.mean([len(v) for v in m.values()]):.2f}")


if __name__ == "__main__":
    main()
