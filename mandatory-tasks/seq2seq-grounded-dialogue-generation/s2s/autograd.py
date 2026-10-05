"""Minimal reverse-mode automatic differentiation in NumPy with fused sequence ops.

Why fused ops: a Python-level autograd node per scalar operation would be far too slow for LSTMs/attention, so the
expensive pieces (LSTM cell, additive/dot attention, softmax cross-entropy) are single nodes with hand-written
backward passes. All of them are verified against finite differences in tests/test_autograd.py.

Conventions: B batch, T time, S source length, H hidden. LSTM state is packed as hc = [h | c] with shape (B, 2H).
"""
from __future__ import annotations

import sys

import numpy as np

DTYPE = np.float32
_GRAD_ENABLED = True


class no_grad:
    """Context manager: ops return plain tensors without building the graph (fast inference)."""

    def __enter__(self):
        global _GRAD_ENABLED
        self.prev, _GRAD_ENABLED = _GRAD_ENABLED, False

    def __exit__(self, *a):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev


class Tensor:
    __slots__ = ("data", "grad", "parents", "backward_fn", "requires_grad", "aux", "name")

    def __init__(self, data, requires_grad=False, parents=(), backward_fn=None, name=""):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.parents = parents
        self.backward_fn = backward_fn
        self.aux = None
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def backward(self, grad=None):
        """Back-propagate from this tensor (scalar loss by default)."""
        order, seen = [], set()
        stack = [(self, False)]
        while stack:                                   # iterative DFS: no recursion limit for long unrolled graphs
            t, done = stack.pop()
            if done:
                order.append(t)
                continue
            if id(t) in seen or not t.requires_grad:
                continue
            seen.add(id(t))
            stack.append((t, True))
            for p in t.parents:
                stack.append((p, False))
        self.grad = np.ones_like(self.data) if grad is None else grad
        for t in reversed(order):
            if t.backward_fn is not None and t.grad is not None:
                t.backward_fn(t.grad)


def param(data, name=""):
    return Tensor(np.asarray(data, dtype=DTYPE), requires_grad=True, name=name)


def const(data):
    return Tensor(np.asarray(data), requires_grad=False)


def _acc(t, g):
    """Accumulate gradient g into tensor t (no-op for constants)."""
    if not t.requires_grad:
        return
    if t.grad is None:
        t.grad = np.array(g, dtype=t.data.dtype, copy=True)
    else:
        t.grad += g


def _make(data, parents, backward_fn):
    rg = _GRAD_ENABLED and any(p.requires_grad for p in parents)
    return Tensor(data, requires_grad=rg, parents=parents if rg else (), backward_fn=backward_fn if rg else None)


def _t(x):
    return x if isinstance(x, Tensor) else const(x)


# ----------------------------------------------------------------------------- basic ops
def embedding(E: Tensor, idx: np.ndarray) -> Tensor:
    """Row lookup: E (V, D), idx int array of any shape -> (*idx.shape, D)."""
    out = E.data[idx]

    def bw(g):
        if E.grad is None:
            E.grad = np.zeros_like(E.data)
        np.add.at(E.grad, idx.reshape(-1), g.reshape(-1, E.data.shape[1]))

    return _make(out, (E,), bw)


def linear(x: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor:
    """x (..., in) @ W (in, out) + b."""
    x = _t(x)
    shp = x.data.shape
    x2 = x.data.reshape(-1, shp[-1])
    out = x2 @ W.data
    if b is not None:
        out = out + b.data
    out = out.reshape(*shp[:-1], W.data.shape[1])
    parents = (x, W) if b is None else (x, W, b)

    def bw(g):
        g2 = g.reshape(-1, g.shape[-1])
        _acc(W, x2.T @ g2)
        if b is not None:
            _acc(b, g2.sum(0))
        _acc(x, (g2 @ W.data.T).reshape(shp))

    return _make(out, parents, bw)


def tanh(x: Tensor) -> Tensor:
    y = np.tanh(x.data)
    return _make(y, (x,), lambda g: _acc(x, g * (1 - y * y)))


def add(a: Tensor, b: Tensor) -> Tensor:
    return _make(a.data + b.data, (a, b), lambda g: (_acc(a, g), _acc(b, g)))


def concat(tensors, axis=-1) -> Tensor:
    tensors = [_t(t) for t in tensors]
    out = np.concatenate([t.data for t in tensors], axis=axis)
    sizes = [t.data.shape[axis] for t in tensors]

    def bw(g):
        start = 0
        for t, s in zip(tensors, sizes):
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(start, start + s)
            _acc(t, g[tuple(sl)])
            start += s

    return _make(out, tuple(tensors), bw)


def dropout(x: Tensor, p: float, rng, training: bool) -> Tensor:
    if not training or p <= 0 or not _GRAD_ENABLED:
        return x
    m = (rng.random(x.data.shape) >= p).astype(x.data.dtype) / (1.0 - p)
    return _make(x.data * m, (x,), lambda g: _acc(x, g * m))


def time_slices(x: Tensor):
    """(B, T, D) -> list of T tensors (B, D); gradients are scattered back into x."""
    T = x.data.shape[1]
    outs = []
    for t in range(T):
        def bw(g, t=t):
            if x.grad is None:
                x.grad = np.zeros_like(x.data)
            x.grad[:, t] += g
        outs.append(_make(x.data[:, t], (x,), bw))
    return outs


def stack_time(tensors) -> Tensor:
    """list of T tensors (B, D) -> (B, T, D)."""
    out = np.stack([t.data for t in tensors], axis=1)

    def bw(g):
        for i, t in enumerate(tensors):
            _acc(t, g[:, i])

    return _make(out, tuple(tensors), bw)


def stack_h(states, H: int) -> Tensor:
    """list of T packed states hc (B, 2H) -> hidden outputs (B, T, H)."""
    out = np.stack([s.data[:, :H] for s in states], axis=1)

    def bw(g):
        for i, s in enumerate(states):
            gg = np.zeros((g.shape[0], 2 * H), dtype=g.dtype)
            gg[:, :H] = g[:, i]
            _acc(s, gg)

    return _make(out, tuple(states), bw)


def reshape(x: Tensor, shape) -> Tensor:
    old = x.data.shape
    return _make(x.data.reshape(shape), (x,), lambda g: _acc(x, g.reshape(old)))


def hidden(hc: Tensor, H: int) -> Tensor:
    """h part of a packed LSTM state (B, 2H) -> (B, H)."""
    def bw(g):
        gg = np.zeros_like(hc.data)
        gg[:, :H] = g
        _acc(hc, gg)
    return _make(hc.data[:, :H], (hc,), bw)


def split_hc(hc: Tensor, H: int):
    """Packed state -> (h, c) tensors."""
    def mk(sl):
        def bw(g):
            gg = np.zeros_like(hc.data)
            gg[:, sl] = g
            _acc(hc, gg)
        return _make(hc.data[:, sl], (hc,), bw)
    return mk(slice(0, H)), mk(slice(H, 2 * H))


# ----------------------------------------------------------------------------- LSTM cell
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lstm_step(x: Tensor, hc_prev: Tensor, Wx: Tensor, Wh: Tensor, b: Tensor, mask=None) -> Tensor:
    """One LSTM step (gate order i, f, g, o). mask (B,1) in {0,1}: where 0 the state is carried over unchanged."""
    H = Wh.data.shape[0]
    h0, c0 = hc_prev.data[:, :H], hc_prev.data[:, H:]
    z = x.data @ Wx.data + h0 @ Wh.data + b.data
    i, f, g, o = _sigmoid(z[:, :H]), _sigmoid(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), _sigmoid(z[:, 3 * H:])
    c1 = f * c0 + i * g
    tc = np.tanh(c1)
    h1 = o * tc
    if mask is None:
        out = np.concatenate([h1, c1], axis=1)
    else:
        out = np.concatenate([mask * h1 + (1 - mask) * h0, mask * c1 + (1 - mask) * c0], axis=1)

    def bw(g_out):
        dh, dc = g_out[:, :H], g_out[:, H:]
        if mask is not None:
            pass_h, pass_c = (1 - mask) * dh, (1 - mask) * dc
            dh, dc = mask * dh, mask * dc
        else:
            pass_h = pass_c = 0.0
        do = dh * tc
        dct = dc + dh * o * (1 - tc * tc)
        di, df, dg = dct * g, dct * c0, dct * i
        dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g * g), do * o * (1 - o)], axis=1)
        _acc(Wx, x.data.T @ dz)
        _acc(Wh, h0.T @ dz)
        _acc(b, dz.sum(0))
        _acc(x, dz @ Wx.data.T)
        _acc(hc_prev, np.concatenate([dz @ Wh.data.T + pass_h, dct * f + pass_c], axis=1))

    return _make(out, (x, hc_prev, Wx, Wh, b), bw)


# ----------------------------------------------------------------------------- attention
def _softmax_masked(scores, mask):
    s = np.where(mask > 0, scores, -1e9)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s) * (mask > 0)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def dot_attention(q: Tensor, enc: Tensor, mask: np.ndarray) -> Tensor:
    """Luong-style dot-product attention. q (B,D), enc (B,S,D), mask (B,S) -> context (B,D).
    The attention weights are stored in the returned tensor's `.aux`."""
    scores = np.einsum("bd,bsd->bs", q.data, enc.data)
    alpha = _softmax_masked(scores, mask)
    ctx = np.einsum("bs,bsd->bd", alpha, enc.data)

    def bw(g):
        dalpha = np.einsum("bd,bsd->bs", g, enc.data)
        dscore = alpha * (dalpha - (alpha * dalpha).sum(1, keepdims=True))
        _acc(q, np.einsum("bs,bsd->bd", dscore, enc.data))
        _acc(enc, alpha[:, :, None] * g[:, None, :] + dscore[:, :, None] * q.data[:, None, :])

    out = _make(ctx, (q, enc), bw)
    out.aux = alpha
    return out


def additive_attention(qp: Tensor, kp: Tensor, v: Tensor, enc: Tensor, mask: np.ndarray) -> Tensor:
    """Bahdanau additive attention: score = v . tanh(qp + kp).
    qp (B,A) projected query, kp (B,S,A) projected keys, v (A,), enc (B,S,De) values -> context (B,De)."""
    e = np.tanh(qp.data[:, None, :] + kp.data)
    scores = e @ v.data
    alpha = _softmax_masked(scores, mask)
    ctx = np.einsum("bs,bsd->bd", alpha, enc.data)

    def bw(g):
        dalpha = np.einsum("bd,bsd->bs", g, enc.data)
        dscore = alpha * (dalpha - (alpha * dalpha).sum(1, keepdims=True))
        dpre = (dscore[:, :, None] * v.data) * (1 - e * e)
        _acc(v, np.einsum("bs,bsa->a", dscore, e))
        _acc(qp, dpre.sum(1))
        _acc(kp, dpre)
        _acc(enc, alpha[:, :, None] * g[:, None, :])

    out = _make(ctx, (qp, kp, v, enc), bw)
    out.aux = alpha
    return out


# ----------------------------------------------------------------------------- loss
def cross_entropy(logits: Tensor, targets: np.ndarray, weights: np.ndarray, label_smoothing: float = 0.0):
    """Mean token-level cross-entropy over positions with weight 1 (padding has weight 0).
    logits (N,V), targets (N,), weights (N,). Returns scalar Tensor; `.aux` = (sum_nll, n_tokens)."""
    z = logits.data.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(z).sum(axis=1))
    logp = z - logZ[:, None]
    N, V = logp.shape
    nll = -logp[np.arange(N), targets]
    if label_smoothing > 0:
        smooth = -logp.mean(axis=1)
        tok = (1 - label_smoothing) * nll + label_smoothing * smooth
    else:
        tok = nll
    n = max(float(weights.sum()), 1.0)
    loss = float((tok * weights).sum() / n)

    def bw(g):
        p = np.exp(logp)
        tgt = np.zeros_like(p)
        tgt[np.arange(N), targets] = 1.0
        if label_smoothing > 0:
            tgt = (1 - label_smoothing) * tgt + label_smoothing / V
        _acc(logits, ((p - tgt) * (weights[:, None] / n) * float(g)).astype(logits.data.dtype))

    out = _make(np.array(loss, dtype=np.float64), (logits,), bw)
    out.aux = (float((nll * weights).sum()), n)
    return out
