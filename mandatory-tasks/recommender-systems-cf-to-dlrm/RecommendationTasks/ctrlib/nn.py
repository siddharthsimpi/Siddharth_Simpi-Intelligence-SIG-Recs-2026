"""Tiny neural-network toolkit with hand-written forward/backward passes (NumPy only).

Everything needed by the CTR models: Param, Linear, ReLU, Dropout, MLP, concatenated embedding tables, Adam.
Default dtype is float32 (fast); tests switch `nn.DTYPE` to float64 for finite-difference gradient checks.
"""
from __future__ import annotations

import numpy as np

DTYPE = np.float32


class Param:
    """A trainable array plus its gradient. `l2` is a coupled L2 (weight-decay) coefficient applied by the optimiser."""

    def __init__(self, value: np.ndarray, name: str, l2: float = 0.0):
        self.value = np.asarray(value, dtype=DTYPE)
        self.grad = np.zeros_like(self.value)
        self.name, self.l2 = name, l2

    def zero_grad(self):
        self.grad.fill(0.0)


class Linear:
    def __init__(self, n_in, n_out, rng, name="linear", weight_decay=0.0, bias=True, std=None):
        std = np.sqrt(2.0 / n_in) if std is None else std          # He init (ReLU networks)
        self.W = Param(rng.normal(0, std, (n_in, n_out)), f"{name}.W", weight_decay)
        self.b = Param(np.zeros(n_out), f"{name}.b") if bias else None

    def forward(self, x, training=True):
        self.x = x
        out = x @ self.W.value
        return out + self.b.value if self.b is not None else out

    def backward(self, g):
        self.W.grad += self.x.T @ g
        if self.b is not None:
            self.b.grad += g.sum(0)
        return g @ self.W.value.T

    def params(self):
        return [self.W] + ([self.b] if self.b is not None else [])


class ReLU:
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, g):
        return g * self.mask

    def params(self):
        return []


class Dropout:
    """Inverted dropout (scales by 1/(1-p) at train time, identity at eval time)."""

    def __init__(self, p, rng):
        self.p, self.rng = p, rng

    def forward(self, x, training=True):
        if not training or self.p <= 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p).astype(DTYPE) / (1.0 - self.p)
        return x * self.mask

    def backward(self, g):
        return g if self.mask is None else g * self.mask

    def params(self):
        return []


class MLP:
    """Fully-connected stack. Hidden layers: Linear -> ReLU -> Dropout. The last layer has a ReLU only if final_relu."""

    def __init__(self, n_in, sizes, rng, dropout=0.0, final_relu=True, weight_decay=0.0, name="mlp"):
        self.ops, self.sizes = [], list(sizes)
        prev = n_in
        for k, s in enumerate(sizes):
            last = k == len(sizes) - 1
            self.ops.append(Linear(prev, s, rng, f"{name}.{k}", weight_decay,
                                   std=(np.sqrt(1.0 / prev) if (last and not final_relu) else None)))
            if not last or final_relu:
                self.ops.append(ReLU())
            if not last and dropout > 0:
                self.ops.append(Dropout(dropout, rng))
            prev = s

    def forward(self, x, training=True):
        for op in self.ops:
            x = op.forward(x, training)
        return x

    def backward(self, g):
        for op in reversed(self.ops):
            g = op.backward(g)
        return g

    def params(self):
        return [p for op in self.ops for p in op.params()]


class Embeddings:
    """All categorical fields in ONE table (rows of field f start at offsets[f]); output shape (B, F, dim)."""

    def __init__(self, cardinalities, dim, rng, l2=0.0, name="emb", init_std=None):
        self.dim, self.F = dim, len(cardinalities)
        self.offsets = np.concatenate([[0], np.cumsum(cardinalities)[:-1]]).astype(np.int64)
        # DLRM reference init: U(-sqrt(1/n_rows), +sqrt(1/n_rows)) per field
        # (init_std given -> N(0, init_std^2) instead)
        if init_std is None:
            blocks = [rng.uniform(-np.sqrt(1.0 / n), np.sqrt(1.0 / n), (n, dim)) for n in cardinalities]
        else:
            blocks = [rng.normal(0, init_std, (n, dim)) for n in cardinalities]
        self.table = Param(np.concatenate(blocks, axis=0), f"{name}.table")
        self.l2 = l2          # sparse L2: only rows used in the batch are regularised

    def forward(self, cat, training=True):
        self.idx = (cat + self.offsets[None, :]).ravel()
        self.B = cat.shape[0]
        return self.table.value[self.idx].reshape(self.B, self.F, self.dim)

    def backward(self, g):
        flat = g.reshape(-1, self.dim)
        np.add.at(self.table.grad, self.idx, flat)
        if self.l2 > 0:
            np.add.at(self.table.grad, self.idx, self.l2 * self.table.value[self.idx])
        return None

    def params(self):
        return [self.table]


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, betas[0], betas[1], eps, 0
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad + p.l2 * p.value if p.l2 > 0 else p.grad
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p.value -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def bce_with_logits(logits, y):
    """Mean binary cross-entropy and d(loss)/d(logit) (numerically stable)."""
    z = logits.astype(np.float64)
    loss = np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z))))
    p = 1.0 / (1.0 + np.exp(-z))
    return float(loss), ((p - y) / len(y)).astype(DTYPE)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))
