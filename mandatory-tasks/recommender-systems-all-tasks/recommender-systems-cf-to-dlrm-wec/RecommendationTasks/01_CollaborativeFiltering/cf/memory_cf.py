"""Memory-based (neighbourhood) collaborative filtering: user-user and item-item kNN.

Similarity   : mean-centred cosine on co-rated entries (Pearson-style; = adjusted cosine for item-item),
               multiplied by a significance-shrinkage factor n_co / (n_co + shrink) so that pairs with
               very few co-ratings are not trusted. Only positive similarities are kept as neighbours.
Prediction   : r_hat(u,i) = mean_u + sum_n s(.,n) * (r_n - mean_n') / sum_n |s(.,n)|   (see formulas below).
Implementation note: neighbours are the top-k most similar users/items *globally* (a sparsified similarity
matrix), not the top-k among those who rated the target. This makes the whole prediction matrix a pair of
matrix products, which is fast, but slightly weaker than the textbook per-target selection.
"""
from __future__ import annotations

import numpy as np


def weighted_average_prediction(similarities, neighbour_ratings) -> float:
    """sum(s*r)/sum(|s|) -- the worked example from the task README gives ((.9*4)+(.8*5))/(.9+.8)=4.47."""
    s = np.asarray(similarities, float)
    r = np.asarray(neighbour_ratings, float)
    return float((s * r).sum() / np.abs(s).sum())


class KNNCF:
    def __init__(self, mode: str = "item", k: int = 50, shrink: float = 25.0,
                 min_rating: float = 1.0, max_rating: float = 5.0):
        assert mode in ("user", "item")
        self.mode, self.k, self.shrink = mode, k, shrink
        self.min_rating, self.max_rating = min_rating, max_rating

    # ------------------------------------------------------------------ fit
    def fit(self, R: np.ndarray, M: np.ndarray) -> "KNNCF":
        self.global_mean = R[M].mean()
        cnt = M.sum(1)
        self.user_mean = np.where(cnt > 0, R.sum(1) / np.maximum(cnt, 1), self.global_mean)
        Rc = (R - self.user_mean[:, None]) * M            # centred ratings, 0 where unobserved
        Mf = M.astype(np.float64)
        self.Rc, self.Mf = Rc, Mf

        X, Mx = (Rc, Mf) if self.mode == "user" else (Rc.T, Mf.T)   # rows = entities being compared
        num = X @ X.T                                               # sum of products on co-rated entries
        A = (X ** 2) @ Mx.T                                         # sum of squares on co-rated entries
        denom = np.sqrt(A * A.T)
        sim = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
        n_co = Mx @ Mx.T
        sim *= np.divide(n_co, n_co + self.shrink, out=np.zeros_like(n_co), where=(n_co + self.shrink) > 0)
        np.fill_diagonal(sim, 0.0)
        sim = np.maximum(sim, 0.0)                                  # positive neighbours only
        self.S = self._top_k(sim, self.k)
        self._pred = None
        return self

    @staticmethod
    def _top_k(sim: np.ndarray, k: int) -> np.ndarray:
        n = sim.shape[0]
        if k >= n:
            return sim
        idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        out = np.zeros_like(sim)
        rows = np.arange(n)[:, None]
        out[rows, idx] = sim[rows, idx]
        return out

    # ------------------------------------------------------------------ predict
    def predict_all(self) -> np.ndarray:
        """Predicted rating for every (user, item); cached."""
        if self._pred is None:
            if self.mode == "user":      # mean_u + sum_v s(u,v)(r_vi - mean_v) / sum_v s(u,v) [v rated i]
                num, den = self.S @ self.Rc, self.S @ self.Mf
                base = self.user_mean[:, None]
            else:                        # mean_u + sum_j s(i,j)(r_uj - mean_u) / sum_j s(i,j) [u rated j]
                num, den = self.Rc @ self.S.T, self.Mf @ self.S.T
                base = self.user_mean[:, None]
            adj = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)   # 0 => fall back to user mean
            self._pred = np.clip(base + adj, self.min_rating, self.max_rating)
        return self._pred

    def predict_pairs(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        return self.predict_all()[users, items]
