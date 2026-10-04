import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from s2s import autograd as ag  # noqa: E402

RNG = np.random.default_rng(0)


def P(*shape, scale=0.5):
    return ag.Tensor(RNG.normal(0, scale, shape).astype(np.float64), requires_grad=True)


def check(build, params, eps=1e-6, tol=1e-6, n=10):
    """build() -> scalar Tensor loss (fresh graph). Compare analytic grads with central differences."""
    for p in params:
        p.grad = None
    loss = build()
    loss.backward()
    worst = 0.0
    for p in params:
        g = p.grad if p.grad is not None else np.zeros_like(p.data)
        flat, gf = p.data.reshape(-1), g.reshape(-1)
        for k in RNG.choice(flat.size, size=min(n, flat.size), replace=False):
            old = flat[k]
            flat[k] = old + eps; lp = float(build().data)
            flat[k] = old - eps; lm = float(build().data)
            flat[k] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - gf[k]) / max(1e-4, abs(num) + abs(gf[k])))
    assert worst < tol, f"relative gradient error {worst:.2e}"


def scalar(t, w):
    """Reduce any tensor to a scalar loss with fixed random weights (so every output element matters)."""
    out = ag._make(np.array((t.data * w).sum()), (t,), None)
    if out.requires_grad:
        out.backward_fn = lambda g: ag._acc(t, w * float(g))
    return out


def test_linear_tanh_concat_embedding():
    E, W, b = P(7, 4), P(8, 5), P(5)
    idx = RNG.integers(0, 7, (3, 2))
    w = RNG.normal(size=(3, 5))
    def build():
        x = ag.concat([ag.embedding(E, idx[:, 0]), ag.embedding(E, idx[:, 1])], axis=-1)
        return scalar(ag.tanh(ag.linear(x, W, b)), w)
    check(build, [E, W, b])


def test_lstm_step_with_mask_and_chain():
    B, D, H, T = 4, 3, 5, 4
    Wx, Wh, b, x0 = P(D, 4 * H), P(H, 4 * H), P(4 * H), P(B, T, D)
    h0 = P(B, 2 * H)
    mask = (RNG.random((B, T)) > 0.3).astype(float)
    w = RNG.normal(size=(B, T, H))
    def build():
        xs = ag.time_slices(x0)
        hc, states = h0, []
        for t in range(T):
            hc = ag.lstm_step(xs[t], hc, Wx, Wh, b, mask[:, t:t + 1])
            states.append(hc)
        return scalar(ag.stack_h(states, H), w)
    check(build, [Wx, Wh, b, x0, h0])


def test_dot_attention():
    B, S, D = 3, 5, 4
    q, enc = P(B, D), P(B, S, D)
    mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], float)
    w = RNG.normal(size=(B, D))
    check(lambda: scalar(ag.dot_attention(q, enc, mask), w), [q, enc])
    out = ag.dot_attention(q, enc, mask)
    assert np.allclose(out.aux.sum(1), 1) and (out.aux[mask == 0] == 0).all()


def test_additive_attention():
    B, S, A, De = 3, 5, 4, 6
    qp, kp, v, enc = P(B, A), P(B, S, A), P(A), P(B, S, De)
    mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], float)
    w = RNG.normal(size=(B, De))
    check(lambda: scalar(ag.additive_attention(qp, kp, v, enc, mask), w), [qp, kp, v, enc])


def test_cross_entropy_and_smoothing():
    logits = P(6, 9)
    tg = RNG.integers(0, 9, 6)
    wt = np.array([1, 1, 0, 1, 0, 1], float)
    check(lambda: ag.cross_entropy(logits, tg, wt), [logits])
    check(lambda: ag.cross_entropy(logits, tg, wt, label_smoothing=0.1), [logits])


def test_no_grad_builds_no_graph():
    W = P(3, 3)
    with ag.no_grad():
        y = ag.linear(P(2, 3), W)
    assert not y.requires_grad and y.parents == ()


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("PASS", k)
