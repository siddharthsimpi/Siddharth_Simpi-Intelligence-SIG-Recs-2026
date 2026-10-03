"""CTR models with explicit forward/backward: LogReg, FM, vanilla MLP, DCN (Deep & Cross), DLRM.

Common interface:   logits = model.forward(dense, cat, training)   # (B,)
                    model.backward(dlogits)                        # accumulate grads into Param.grad
                    model.params()                                 # list[Param]
`dense` is (B, n_dense) float, `cat` is (B, 26) integer ids (0 = rare/unseen bucket).
"""
from __future__ import annotations

import numpy as np

from . import nn
from .nn import MLP, Embeddings, Linear, Param


class Schema:
    def __init__(self, n_dense, cardinalities):
        self.n_dense, self.cardinalities, self.n_cat = n_dense, list(cardinalities), len(cardinalities)


class BaseModel:
    def n_params(self):
        return int(sum(p.value.size for p in self.params()))

    def param_breakdown(self):
        emb = sum(p.value.size for p in self.params() if p.name.endswith("table"))
        return {"embedding_params": int(emb), "dense_params": int(self.n_params() - emb)}


# --------------------------------------------------------------------------- 0) Logistic regression
class LogReg(BaseModel):
    """logit = b + w.dense + sum_f w_f[cat_f]   (one scalar weight per category)."""

    def __init__(self, schema, seed=0, weight_decay=0.0, emb_l2=0.0, **_):
        rng = np.random.default_rng(seed)
        self.dense = Linear(schema.n_dense, 1, rng, "lr.dense", weight_decay, std=0.01)
        self.lin = Embeddings(schema.cardinalities, 1, rng, emb_l2, "lr.lin")
        self.lin.table.value *= 0.0

    def forward(self, dense, cat, training=True):
        self.B, self.F = len(cat), cat.shape[1]
        return self.dense.forward(dense)[:, 0] + self.lin.forward(cat)[:, :, 0].sum(1)

    def backward(self, g):
        self.dense.backward(g[:, None])
        self.lin.backward(np.broadcast_to(g[:, None, None], (self.B, self.F, 1)).copy())

    def params(self):
        return self.dense.params() + self.lin.params()


# --------------------------------------------------------------------------- 1) Factorization machine
class FM(BaseModel):
    """Linear part + sum_{i<j} <v_i, v_j> over all fields (26 categorical + 1 'dense' field = Linear(dense)).

    This is matrix factorization generalised from (user, item) to many fields: with two ID fields it reduces
    to biased MF  b_u + b_i + <p_u, q_i>  -- the model family of Task 1.
    """

    def __init__(self, schema, emb_dim=8, seed=0, weight_decay=0.0, emb_l2=0.0, emb_init=None, **_):
        rng = np.random.default_rng(seed)
        self.linear = LogReg(schema, seed, weight_decay, emb_l2)
        self.emb = Embeddings(schema.cardinalities, emb_dim, rng, emb_l2, "fm.emb", emb_init)
        self.dproj = Linear(schema.n_dense, emb_dim, rng, "fm.dproj", weight_decay, bias=False, std=0.1)

    def forward(self, dense, cat, training=True):
        e = self.emb.forward(cat)
        self.Z = np.concatenate([self.dproj.forward(dense)[:, None, :], e], axis=1)     # (B, F+1, d)
        self.s = self.Z.sum(1)
        inter = 0.5 * ((self.s ** 2).sum(-1) - (self.Z ** 2).sum((1, 2)))
        return self.linear.forward(dense, cat) + inter

    def backward(self, g):
        self.linear.backward(g)
        dZ = g[:, None, None] * (self.s[:, None, :] - self.Z)
        self.dproj.backward(dZ[:, 0])
        self.emb.backward(dZ[:, 1:])

    def params(self):
        return self.linear.params() + self.emb.params() + self.dproj.params()


# --------------------------------------------------------------------------- 2) Vanilla MLP
class MLPModel(BaseModel):
    """concat[dense, flattened embeddings] -> hidden ReLU layers -> logit."""

    def __init__(self, schema, emb_dim=16, hidden=(128, 64), dropout=0.0, seed=0, weight_decay=0.0,
                 emb_l2=0.0, emb_init=None, **_):
        rng = np.random.default_rng(seed)
        self.emb = Embeddings(schema.cardinalities, emb_dim, rng, emb_l2, "emb", emb_init)
        self.n_dense = schema.n_dense
        self.net = MLP(schema.n_dense + schema.n_cat * emb_dim, list(hidden) + [1], rng, dropout,
                       final_relu=False, weight_decay=weight_decay, name="mlp")

    def forward(self, dense, cat, training=True):
        e = self.emb.forward(cat)
        self.B = len(cat)
        x = np.concatenate([dense, e.reshape(self.B, -1)], axis=1)
        return self.net.forward(x, training)[:, 0]

    def backward(self, g):
        dx = self.net.backward(g[:, None])
        self.emb.backward(dx[:, self.n_dense:].reshape(self.B, self.emb.F, self.emb.dim))

    def params(self):
        return self.emb.params() + self.net.params()


# --------------------------------------------------------------------------- 3) Deep & Cross Network (bonus)
class DCN(BaseModel):
    """Deep & Cross Network (Wang et al. 2017). x0 = concat[dense, embeddings];
    cross layer: x_{l+1} = x0 * (x_l . w_l) + b_l + x_l  (polynomial degree grows by one per layer);
    deep tower in parallel; logit = Linear(concat[x_L, deep_out])."""

    def __init__(self, schema, emb_dim=8, hidden=(128, 64), n_cross=3, dropout=0.0, seed=0,
                 weight_decay=0.0, emb_l2=0.0, emb_init=None, **_):
        rng = np.random.default_rng(seed)
        self.emb = Embeddings(schema.cardinalities, emb_dim, rng, emb_l2, "emb", emb_init)
        self.n_dense = schema.n_dense
        D = schema.n_dense + schema.n_cat * emb_dim
        self.D, self.n_cross = D, n_cross
        self.cw = [Param(rng.normal(0, np.sqrt(1.0 / D), D), f"dcn.cw{l}", weight_decay) for l in range(n_cross)]
        self.cb = [Param(np.zeros(D), f"dcn.cb{l}") for l in range(n_cross)]
        self.deep = MLP(D, list(hidden), rng, dropout, final_relu=True, weight_decay=weight_decay, name="dcn.deep")
        self.out = Linear(D + hidden[-1], 1, rng, "dcn.out", weight_decay, std=np.sqrt(1.0 / (D + hidden[-1])))

    def forward(self, dense, cat, training=True):
        e = self.emb.forward(cat)
        self.B = len(cat)
        self.x0 = np.concatenate([dense, e.reshape(self.B, -1)], axis=1)
        self.xs, self.ss = [self.x0], []
        x = self.x0
        for w, b in zip(self.cw, self.cb):
            s = (x @ w.value)[:, None]                       # (B,1)
            x = self.x0 * s + b.value + x
            self.ss.append(s); self.xs.append(x)
        h = self.deep.forward(self.x0, training)
        return self.out.forward(np.concatenate([x, h], axis=1))[:, 0]

    def backward(self, g):
        dcat = self.out.backward(g[:, None])
        gx, gh = dcat[:, :self.D], dcat[:, self.D:]
        dx0 = self.deep.backward(gh)
        for l in reversed(range(self.n_cross)):
            xl, s = self.xs[l], self.ss[l]
            t = (gx * self.x0).sum(1, keepdims=True)         # dL/ds
            self.cw[l].grad += (xl * t).sum(0)
            self.cb[l].grad += gx.sum(0)
            dx0 = dx0 + gx * s
            gx = gx + t * self.cw[l].value[None, :]
        dx0 = dx0 + gx                                       # x_0 = x0
        self.emb.backward(dx0[:, self.n_dense:].reshape(self.B, self.emb.F, self.emb.dim))

    def params(self):
        return self.emb.params() + self.cw + self.cb + self.deep.params() + self.out.params()


# --------------------------------------------------------------------------- 4) DLRM
class DLRM(BaseModel):
    """Deep Learning Recommendation Model (Naumov et al. 2019).

    dense --bottom MLP--> z0 (dim d);  26 categorical --embedding tables--> e_1..e_26 (dim d)
    interaction='dot': all pairwise dot products among {z0, e_1..e_26}  (27*26/2 = 351 numbers)
    top MLP input = concat[z0, dot products]  -> top MLP -> logit        (sigmoid applied in the loss)
    interaction='cat': ablation, no explicit interaction: concat[z0, e_1..e_26] -> top MLP.
    """

    def __init__(self, schema, emb_dim=16, bottom=(64,), top=(128, 64), interaction="dot", dropout=0.0,
                 seed=0, weight_decay=0.0, emb_l2=0.0, emb_init=None, **_):
        assert interaction in ("dot", "cat")
        rng = np.random.default_rng(seed)
        self.d, self.interaction, self.F = emb_dim, interaction, schema.n_cat
        self.emb = Embeddings(schema.cardinalities, emb_dim, rng, emb_l2, "emb", emb_init)
        self.bottom = MLP(schema.n_dense, list(bottom) + [emb_dim], rng, 0.0, final_relu=True,
                          weight_decay=weight_decay, name="dlrm.bottom")
        N = schema.n_cat + 1
        self.li, self.lj = np.tril_indices(N, k=-1)
        top_in = emb_dim + len(self.li) if interaction == "dot" else emb_dim * N
        self.top = MLP(top_in, list(top) + [1], rng, dropout, final_relu=False,
                       weight_decay=weight_decay, name="dlrm.top")

    def forward(self, dense, cat, training=True):
        self.B = len(cat)
        z0 = self.bottom.forward(dense, training)                              # (B,d)
        e = self.emb.forward(cat)                                              # (B,F,d)
        self.Z = np.concatenate([z0[:, None, :], e], axis=1)                   # (B,N,d)
        if self.interaction == "dot":
            G = self.Z @ self.Z.transpose(0, 2, 1)                             # (B,N,N) all dot products
            x = np.concatenate([z0, G[:, self.li, self.lj]], axis=1)
        else:
            x = self.Z.reshape(self.B, -1)
        return self.top.forward(x, training)[:, 0]

    def backward(self, g):
        dx = self.top.backward(g[:, None])
        if self.interaction == "dot":
            dz0, dp = dx[:, :self.d], dx[:, self.d:]
            dG = np.zeros((self.B, self.Z.shape[1], self.Z.shape[1]), dtype=nn.DTYPE)
            dG[:, self.li, self.lj] = dp
            dZ = (dG + dG.transpose(0, 2, 1)) @ self.Z                         # dZ_i = sum_j (dG_ij + dG_ji) Z_j
            dZ[:, 0] += dz0
        else:
            dZ = dx.reshape(self.B, self.Z.shape[1], self.d)
        self.bottom.backward(dZ[:, 0])
        self.emb.backward(dZ[:, 1:])

    def params(self):
        return self.emb.params() + self.bottom.params() + self.top.params()


REGISTRY = {"logreg": LogReg, "fm": FM, "mlp": MLPModel, "dcn": DCN, "dlrm": DLRM}


def build_model(spec: dict, schema: Schema, seed: int):
    spec = dict(spec)
    kind = spec.pop("type")
    for k in ("lr", "batch_size", "name"):          # training-loop settings, not model arguments
        spec.pop(k, None)
    return REGISTRY[kind](schema, seed=seed, **spec)
