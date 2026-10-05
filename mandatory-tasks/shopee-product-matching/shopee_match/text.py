"""Text utilities: normalisation, TF-IDF, simple token features."""
import re
import unicodedata

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_NUM = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_title(s) -> str:
    """NFKC -> lowercase -> punctuation to spaces -> collapse whitespace (keeps unicode letters)."""
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def numbers(s: str) -> set:
    return set(_NUM.findall(s))


def tfidf_matrix(titles, analyzer="word", ngram_range=(1, 1), min_df=1):
    """L2-normalised TF-IDF (sublinear tf). Fitted on all titles: unsupervised, uses no labels."""
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=min_df,
                          sublinear_tf=True, lowercase=False, dtype=np.float32)
    return vec.fit_transform(list(titles)).tocsr()


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
