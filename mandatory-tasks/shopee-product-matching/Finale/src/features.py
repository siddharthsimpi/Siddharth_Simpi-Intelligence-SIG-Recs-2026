"""Candidate generation and pair-feature construction for the multimodal matcher."""
import numpy as np
import pandas as pd

from shopee_match.pairs import pair_hamming, phash_u64
from shopee_match.retrieval import pair_scores, topk_pairs
from shopee_match.text import jaccard, normalize_title, numbers


def candidate_pairs(E, mods, rows, k, labels):
    """Stage 1 - RETRIEVAL. Union of the directed top-k neighbours of every listing under each modality.
    Using several modalities for candidate generation raises the recall ceiling (a match missed by
    text can still be retrieved by the image, and vice versa)."""
    n = len(labels)
    keys = []
    for m in mods:
        i, j, _ = topk_pairs(E[m], rows, k)
        keys.append(i.astype(np.int64) * n + j)
    key = np.unique(np.concatenate(keys))
    i, j = key // n, key % n
    out = pd.DataFrame({"i": i, "j": j})
    out["y"] = (labels[i] == labels[j]).astype(np.int8)
    return out


class PairFeaturizer:
    """Stage 2 - PAIR FEATURES. All features are symmetric in (i, j)."""

    def __init__(self, df, E, text_mods, img_mods):
        self.E, self.text_mods, self.img_mods = E, list(text_mods), list(img_mods)
        norm = [normalize_title(t) for t in df["title"]]
        self.nchar = np.array([len(t) for t in norm], dtype=np.float32)
        self.toks = [set(t.split()) for t in norm]
        self.nw = np.array([len(t) for t in self.toks])
        self.nums = [numbers(t) for t in norm]
        self.ph = phash_u64(df["image_phash"])
        self.image = df["image"].values

    @property
    def groups(self):
        return {"text": [f"cos_{m}" for m in self.text_mods],
                "image": [f"cos_{m}" for m in self.img_mods],
                "phash": ["ham", "phash_eq", "same_file"],
                "meta": ["len_ratio", "word_jacc", "num_jacc", "num_both", "num_conflict", "min_words"]}

    def __call__(self, cand):
        i, j = cand["i"].values, cand["j"].values
        F = pd.DataFrame(index=cand.index)
        for m in self.text_mods + self.img_mods:
            F[f"cos_{m}"] = pair_scores(self.E[m], i, j)
        F["ham"] = pair_hamming(self.ph, i, j).astype(np.float32)
        F["phash_eq"] = (F["ham"] == 0).astype(np.float32)
        F["same_file"] = (self.image[i] == self.image[j]).astype(np.float32)
        a, b = self.nchar[i], self.nchar[j]
        F["len_ratio"] = np.minimum(a, b) / np.maximum(np.maximum(a, b), 1)
        F["word_jacc"] = [jaccard(self.toks[x], self.toks[y]) for x, y in zip(i, j)]
        F["num_jacc"] = [jaccard(self.nums[x], self.nums[y]) for x, y in zip(i, j)]
        both = [bool(self.nums[x]) and bool(self.nums[y]) for x, y in zip(i, j)]
        F["num_both"] = np.array(both, dtype=np.float32)
        F["num_conflict"] = np.array([bo and self.nums[x] != self.nums[y] for bo, x, y in zip(both, i, j)], dtype=np.float32)
        F["min_words"] = np.minimum(self.nw[i], self.nw[j]).astype(np.float32)
        return F.astype(np.float32)
