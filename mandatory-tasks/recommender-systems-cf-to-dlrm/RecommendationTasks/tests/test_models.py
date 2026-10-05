import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from ctrlib import nn  # noqa: E402

nn.DTYPE = np.float64      # exact gradient checks
from ctrlib.models import Schema, build_model  # noqa: E402

SPECS = {
    "logreg": {"type": "logreg"},
    "fm": {"type": "fm", "emb_dim": 4},
    "mlp": {"type": "mlp", "emb_dim": 4, "hidden": [8, 6]},
    "dcn": {"type": "dcn", "emb_dim": 4, "hidden": [8, 6], "n_cross": 3},
    "dlrm_dot": {"type": "dlrm", "emb_dim": 4, "bottom": [6], "top": [8, 5], "interaction": "dot"},
    "dlrm_cat": {"type": "dlrm", "emb_dim": 4, "bottom": [6], "top": [8, 5], "interaction": "cat"},
}


def _grad_check(spec, with_reg):
    rng = np.random.default_rng(0)
    schema = Schema(3, [5, 4, 6])
    B = 7
    dense = rng.normal(size=(B, 3))
    cat = np.stack([rng.integers(0, c, B) for c in schema.cardinalities], axis=1)
    y = rng.integers(0, 2, B).astype(float)
    s = dict(spec)
    if with_reg:
        s.update(weight_decay=0.0, emb_l2=0.0)    # regularisers are applied in optimiser / outside the loss
    m = build_model(s, schema, seed=3)
    for p in m.params():                           # make biases / zero-init tables non-trivial
        p.value += rng.normal(0, 0.05, p.value.shape)

    def loss():
        return nn.bce_with_logits(m.forward(dense, cat, training=False), y)[0]

    for p in m.params():
        p.zero_grad()
    _, g = nn.bce_with_logits(m.forward(dense, cat, training=False), y)
    m.backward(g)
    worst = 0.0
    eps = 1e-6
    for p in m.params():
        flat, gflat = p.value.reshape(-1), p.grad.reshape(-1)
        for k in rng.choice(flat.size, size=min(12, flat.size), replace=False):
            old = flat[k]
            flat[k] = old + eps; lp = loss()
            flat[k] = old - eps; lm = loss()
            flat[k] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - gflat[k]) / max(1e-7, abs(num) + abs(gflat[k])))
    return worst


def test_gradients_all_models():
    for name, spec in SPECS.items():
        err = _grad_check(spec, with_reg=False)
        assert err < 1e-5, f"{name}: relative gradient error {err:.2e}"


def test_dlrm_dot_interaction_values():
    schema = Schema(2, [3, 3])
    m = build_model({"type": "dlrm", "emb_dim": 2, "bottom": [3], "top": [4], "interaction": "dot"}, schema, 0)
    dense = np.array([[0.5, -1.0]])
    cat = np.array([[1, 2]])
    m.forward(dense, cat, training=False)
    Z = m.Z[0]                                   # (3, d): z0, e1, e2
    expected = [Z[1] @ Z[0], Z[2] @ Z[0], Z[2] @ Z[1]]   # lower-triangle order (1,0),(2,0),(2,1)
    got = (Z @ Z.T)[m.li, m.lj]
    assert np.allclose(got, expected) and len(got) == 3


def test_param_counts_dlrm():
    schema = Schema(13, [10] * 26)
    m = build_model({"type": "dlrm", "emb_dim": 16, "bottom": [64], "top": [128, 64]}, schema, 0)
    emb = 260 * 16
    bottom = 13 * 64 + 64 + 64 * 16 + 16
    top_in = 16 + 27 * 26 // 2
    top = top_in * 128 + 128 + 128 * 64 + 64 + 64 + 1
    assert m.n_params() == emb + bottom + top


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("PASS", k)
