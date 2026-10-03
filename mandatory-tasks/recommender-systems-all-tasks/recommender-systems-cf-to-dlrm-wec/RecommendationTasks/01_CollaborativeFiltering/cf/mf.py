"""Biased matrix factorization trained from scratch with NumPy (mini-batch Adam or plain SGD).

    r_hat(u,i) = mu + b_u + b_i + p_u . q_i
    loss       = mean over batch of [ (r - r_hat)^2 + lam * (|p_u|^2 + |q_i|^2 + b_u^2 + b_i^2) ]

* Adam : per-parameter adaptive step sizes -> far less learning-rate tuning and fast convergence, which matters
         because rare users/items get tiny, noisy gradients. Cost: 2 extra state arrays per parameter and
         sometimes slightly worse generalisation than well-tuned SGD.
* SGD  : plain per-sample-equivalent updates; simplest, lowest memory, but sensitive to the learning rate.
Weights are initialised N(0, 0.1^2); no pretrained weights are used. Early stopping on validation RMSE.
"""
from __future__ import annotations

import numpy as np


class BiasedMF:
    def __init__(self, n_users: int, n_items: int, n_factors: int = 32, lr: float = 0.01, reg: float = 0.05,
                 batch_size: int = 1024, epochs: int = 60, patience: int = 5, optimizer: str = "adam",
                 init_std: float = 0.1, seed: int = 0, min_rating: float = 1.0, max_rating: float = 5.0):
        assert optimizer in ("adam", "sgd")
        self.n_users, self.n_items, self.n_factors = n_users, n_items, n_factors
        self.lr, self.reg, self.batch_size, self.epochs, self.patience = lr, reg, batch_size, epochs, patience
        self.optimizer, self.seed = optimizer, seed
        self.min_rating, self.max_rating = min_rating, max_rating
        rng = np.random.default_rng(seed)
        self.params = {
            "P": rng.normal(0, init_std, (n_users, n_factors)),
            "Q": rng.normal(0, init_std, (n_items, n_factors)),
            "bu": np.zeros(n_users),
            "bi": np.zeros(n_items),
        }
        self.mu = 0.0
        self.history = {"epoch": [], "train_rmse": [], "val_rmse": []}
        self.best_epoch = 0

    # ------------------------------------------------------------------ model
    def predict_pairs(self, u, i, clip: bool = True) -> np.ndarray:
        p = self.params
        pred = self.mu + p["bu"][u] + p["bi"][i] + (p["P"][u] * p["Q"][i]).sum(1)
        return np.clip(pred, self.min_rating, self.max_rating) if clip else pred

    def predict_all(self, clip: bool = True) -> np.ndarray:
        p = self.params
        pred = self.mu + p["bu"][:, None] + p["bi"][None, :] + p["P"] @ p["Q"].T
        return np.clip(pred, self.min_rating, self.max_rating) if clip else pred

    def loss_and_grads(self, u, i, r):
        """Mean regularised squared loss on a batch and its analytic gradients (used by the gradient-check test)."""
        p = self.params
        B = len(r)
        Pu, Qi = p["P"][u], p["Q"][i]
        err = self.mu + p["bu"][u] + p["bi"][i] + (Pu * Qi).sum(1) - r
        reg = self.reg
        loss = np.mean(err ** 2 + reg * ((Pu ** 2).sum(1) + (Qi ** 2).sum(1) + p["bu"][u] ** 2 + p["bi"][i] ** 2))
        g = {k: np.zeros_like(v) for k, v in p.items()}
        np.add.at(g["P"], u, (2 * err[:, None] * Qi + 2 * reg * Pu) / B)
        np.add.at(g["Q"], i, (2 * err[:, None] * Pu + 2 * reg * Qi) / B)
        np.add.at(g["bu"], u, (2 * err + 2 * reg * p["bu"][u]) / B)
        np.add.at(g["bi"], i, (2 * err + 2 * reg * p["bi"][i]) / B)
        return loss, g

    # ------------------------------------------------------------------ optimisation
    def _step(self, grads, B):
        if self.optimizer == "sgd":
            for k, g in grads.items():
                self.params[k] -= self.lr * B * g          # B * mean-gradient = sum of per-sample gradients
            return
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * g * g
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            self.params[k] -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def fit(self, u, i, r, val=None, verbose: bool = False) -> "BiasedMF":
        u, i, r = np.asarray(u), np.asarray(i), np.asarray(r, float)
        self.mu = float(r.mean())
        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._v = {k: np.zeros_like(v) for k, v in self.params.items()}
        rng = np.random.default_rng(self.seed + 1)
        best, best_params, bad = np.inf, None, 0
        n = len(r)
        for epoch in range(1, self.epochs + 1):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                b = perm[s:s + self.batch_size]
                _, grads = self.loss_and_grads(u[b], i[b], r[b])
                self._step(grads, len(b))
            tr = float(np.sqrt(np.mean((self.predict_pairs(u, i) - r) ** 2)))
            self.history["epoch"].append(epoch)
            self.history["train_rmse"].append(tr)
            if val is not None:
                vu, vi, vr = val
                va = float(np.sqrt(np.mean((self.predict_pairs(vu, vi) - vr) ** 2)))
                self.history["val_rmse"].append(va)
                if va < best - 1e-5:
                    best, bad, self.best_epoch = va, 0, epoch
                    best_params = {k: v.copy() for k, v in self.params.items()}
                else:
                    bad += 1
                if verbose:
                    print(f"  epoch {epoch:3d} train {tr:.4f} val {va:.4f}")
                if bad >= self.patience:
                    break
            else:
                self.best_epoch = epoch
        if best_params is not None:
            self.params = best_params                      # restore best-validation weights
        return self
