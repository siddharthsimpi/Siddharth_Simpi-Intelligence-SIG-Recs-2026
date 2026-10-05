"""Score-level fusion and the learned (gradient-boosted) pair classifier."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def weighted_score(F, text_cols, img_cols, w_text):
    """s = w * mean(text cosines) + (1 - w) * mean(image cosines)."""
    st = F[text_cols].values.mean(1) if text_cols else 0.0
    si = F[img_cols].values.mean(1) if img_cols else 0.0
    if not text_cols:
        return si
    if not img_cols:
        return st
    return w_text * st + (1 - w_text) * si


def fit_gbm(F, y, cols, seed=42, **kw):
    """Learned fusion. Small, regularised GBM: the signal is low-dimensional (<= 20 features) and
    monotone in the similarity features, so a big model would only overfit the train products."""
    params = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=seed)
    params.update(kw)
    return HistGradientBoostingClassifier(**params).fit(F[cols].values, y)


def gbm_score(model, F, cols):
    return model.predict_proba(F[cols].values)[:, 1].astype(np.float32)
