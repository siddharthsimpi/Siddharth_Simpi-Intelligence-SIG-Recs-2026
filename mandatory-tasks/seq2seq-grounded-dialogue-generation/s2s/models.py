"""Encoder-decoder (LSTM) with optional attention and optional multi-source conditioning.

One class covers every experiment:
  Sub-task 1  Seq2Seq(attn="none",   n_sources=1)   classic Sutskever-style: encoder final state -> decoder initial state
  Sub-task 2  Seq2Seq(attn="bahdanau"|"luong_general"|"luong_dot", n_sources=1)
  Sub-task 3  Seq2Seq(attn=..., n_sources=2)        sources = [dialogue history, grounding document]; the decoder attends
              over both encoders' states at every step (contexts are concatenated); n_sources=1 gives the ungrounded baseline.

Attention variants
  bahdanau       additive attention, query = previous decoder state, context feeds the LSTM input (Bahdanau et al. 2015)
  luong_general  bilinear score h_t W h_s, query = current decoder state, input feeding (Luong et al. 2015)
  luong_dot      dot-product score h_t . h_s
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from . import autograd as ag
from .autograd import Tensor


@dataclass
class ModelConfig:
    emb_dim: int = 128
    hid: int = 128
    attn: str = "none"            # none | bahdanau | luong_general | luong_dot
    n_sources: int = 1
    dropout: float = 0.2
    att_dim: int = 128            # size of the additive-attention space
    share_embeddings: bool = False  # one table for all sources and the target (same vocabulary)
    label_smoothing: float = 0.0


def _glorot(rng, n_in, n_out):
    lim = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-lim, lim, (n_in, n_out))


class Seq2Seq:
    def __init__(self, cfg: ModelConfig, src_vocab_sizes, tgt_vocab_size, seed: int = 0):
        assert cfg.attn in ("none", "bahdanau", "luong_general", "luong_dot")
        assert len(src_vocab_sizes) == cfg.n_sources
        self.cfg, self.src_sizes, self.tgt_size = cfg, list(src_vocab_sizes), tgt_vocab_size
        self.rng = np.random.default_rng(seed)
        r, H, E, A = self.rng, cfg.hid, cfg.emb_dim, cfg.att_dim
        p: dict[str, Tensor] = {}

        def add(name, arr):
            p[name] = ag.param(arr, name)

        def lstm(prefix, n_in):
            add(prefix + ".Wx", _glorot(r, n_in, 4 * H))
            add(prefix + ".Wh", _glorot(r, H, 4 * H))
            b = np.zeros(4 * H)
            b[H:2 * H] = 1.0                       # forget-gate bias 1: remember by default
            add(prefix + ".b", b)

        # embeddings
        if cfg.share_embeddings:
            assert len(set(self.src_sizes + [tgt_vocab_size])) == 1, "shared embeddings need one common vocabulary"
            add("emb", r.normal(0, 0.1, (tgt_vocab_size, E)))
            self.emb_src = [p["emb"]] * cfg.n_sources
            self.emb_tgt = p["emb"]
        else:
            self.emb_src = []
            for k, v in enumerate(self.src_sizes):
                add(f"emb_src{k}", r.normal(0, 0.1, (v, E)))
                self.emb_src.append(p[f"emb_src{k}"])
            add("emb_tgt", r.normal(0, 0.1, (tgt_vocab_size, E)))
            self.emb_tgt = p["emb_tgt"]
        # encoders
        for k in range(cfg.n_sources):
            lstm(f"enc{k}", E)
        if cfg.n_sources > 1:                       # bridge: concat of all encoders' final [h|c] -> decoder initial [h|c]
            add("bridge.W", _glorot(r, 2 * H * cfg.n_sources, 2 * H))
            add("bridge.b", np.zeros(2 * H))
        # decoder
        n_ctx = 0 if cfg.attn == "none" else cfg.n_sources
        if cfg.attn == "none":
            d_in = E
        elif cfg.attn == "bahdanau":
            d_in = E + n_ctx * H
        else:
            d_in = E + H                            # input feeding of the previous attentional vector
        lstm("dec", d_in)
        if cfg.attn == "bahdanau":
            for k in range(cfg.n_sources):
                add(f"att{k}.Wq", _glorot(r, H, A))
                add(f"att{k}.Wk", _glorot(r, H, A))
                add(f"att{k}.v", r.normal(0, 0.1, A))
            add("out.Wo", _glorot(r, H + n_ctx * H + E, H))
            add("out.bo", np.zeros(H))
        elif cfg.attn.startswith("luong"):
            if cfg.attn == "luong_general":
                for k in range(cfg.n_sources):
                    add(f"att{k}.Wa", _glorot(r, H, H))
            add("out.Wo", _glorot(r, H + n_ctx * H, H))
            add("out.bo", np.zeros(H))
        add("proj.W", _glorot(r, H, tgt_vocab_size))
        add("proj.b", np.zeros(tgt_vocab_size))
        self.p = p

    # ------------------------------------------------------------------ bookkeeping
    def parameters(self):
        seen, out = set(), []
        for t in self.p.values():
            if id(t) not in seen and t.requires_grad:
                seen.add(id(t)); out.append(t)
        return out

    def n_params(self):
        return int(sum(t.data.size for t in self.parameters()))

    def zero_grad(self):
        for t in self.p.values():
            t.grad = None

    def state_dict(self):
        return {k: v.data for k, v in self.p.items()}

    def load_state_dict(self, d):
        for k, v in self.p.items():
            v.data[...] = d[k]

    def save(self, path):
        np.savez(path, __cfg__=json.dumps({"cfg": asdict(self.cfg), "src": self.src_sizes, "tgt": self.tgt_size}),
                 **self.state_dict())

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["__cfg__"]))
        m = cls(ModelConfig(**meta["cfg"]), meta["src"], meta["tgt"])
        m.load_state_dict({k: z[k] for k in z.files if k != "__cfg__"})
        return m

    def set_embeddings(self, matrices: dict, freeze: bool = False):
        """matrices: {'src0': (V,E), 'tgt': (V,E), 'all': (V,E)} pretrained vectors copied into the tables."""
        tables = {f"src{k}": t for k, t in enumerate(self.emb_src)}
        tables["tgt"] = self.emb_tgt
        for name, mat in matrices.items():
            targets = list(tables.values()) if name == "all" else [tables[name]]
            for t in targets:
                assert t.data.shape == mat.shape, (name, t.data.shape, mat.shape)
                t.data[...] = mat
                if freeze:
                    t.requires_grad = False

    # ------------------------------------------------------------------ encoder
    def encode(self, srcs, training=False):
        """srcs: list (one per source) of (ids (B,S) int, mask (B,S) float). Returns dict with encoder states."""
        cfg, H = self.cfg, self.cfg.hid
        outs, finals, kps = [], [], []
        for k, (ids, mask) in enumerate(srcs):
            B, S = ids.shape
            x = ag.dropout(ag.embedding(self.emb_src[k], ids), cfg.dropout, self.rng, training)
            xs = ag.time_slices(x)
            hc = ag.const(np.zeros((B, 2 * H), dtype=ag.DTYPE))
            states = []
            Wx, Wh, b = self.p[f"enc{k}.Wx"], self.p[f"enc{k}.Wh"], self.p[f"enc{k}.b"]
            for t in range(S):
                hc = ag.lstm_step(xs[t], hc, Wx, Wh, b, mask[:, t:t + 1])
                states.append(hc)
            o = ag.stack_h(states, H)
            outs.append(o)
            finals.append(hc)
            if cfg.attn == "bahdanau":
                kps.append(ag.linear(o, self.p[f"att{k}.Wk"]))        # keys projected once per batch
        if cfg.n_sources == 1:
            hc0 = finals[0]
        else:
            hc0 = ag.linear(ag.concat(finals, axis=1), self.p["bridge.W"], self.p["bridge.b"])
        return {"outs": outs, "masks": [m for _, m in srcs], "kps": kps, "hc0": hc0}

    def init_state(self, enc):
        B = enc["hc0"].data.shape[0]
        a = ag.const(np.zeros((B, self.cfg.hid), dtype=ag.DTYPE)) if self.cfg.attn.startswith("luong") else None
        return {"hc": enc["hc0"], "a": a}

    # ------------------------------------------------------------------ decoder step
    def step(self, state, y_emb: Tensor, enc, training=False):
        """One decoder step. Returns (o (B,H) pre-softmax features, new_state, [attention weights per source])."""
        cfg, H = self.cfg, self.cfg.hid
        P = self.p
        hc, alphas = state["hc"], []
        if cfg.attn == "none":
            hc = ag.lstm_step(y_emb, hc, P["dec.Wx"], P["dec.Wh"], P["dec.b"])
            return ag.hidden(hc, H), {"hc": hc, "a": None}, alphas
        if cfg.attn == "bahdanau":
            h_prev = ag.hidden(hc, H)
            ctxs = []
            for k in range(cfg.n_sources):
                qp = ag.linear(h_prev, P[f"att{k}.Wq"])
                c = ag.additive_attention(qp, enc["kps"][k], P[f"att{k}.v"], enc["outs"][k], enc["masks"][k])
                ctxs.append(c); alphas.append(c.aux)
            hc = ag.lstm_step(ag.concat([y_emb] + ctxs, axis=1), hc, P["dec.Wx"], P["dec.Wh"], P["dec.b"])
            h = ag.hidden(hc, H)
            o = ag.tanh(ag.linear(ag.concat([h] + ctxs + [y_emb], axis=1), P["out.Wo"], P["out.bo"]))
            return o, {"hc": hc, "a": None}, alphas
        # Luong: LSTM first, then attention with the new hidden state, input feeding
        hc = ag.lstm_step(ag.concat([y_emb, state["a"]], axis=1), hc, P["dec.Wx"], P["dec.Wh"], P["dec.b"])
        h = ag.hidden(hc, H)
        ctxs = []
        for k in range(cfg.n_sources):
            q = h if cfg.attn == "luong_dot" else ag.linear(h, P[f"att{k}.Wa"])
            c = ag.dot_attention(q, enc["outs"][k], enc["masks"][k])
            ctxs.append(c); alphas.append(c.aux)
        a = ag.tanh(ag.linear(ag.concat([h] + ctxs, axis=1), P["out.Wo"], P["out.bo"]))
        return a, {"hc": hc, "a": a}, alphas

    # ------------------------------------------------------------------ training loss
    def loss(self, srcs, tgt_in, tgt_out, tgt_mask, training=True):
        """Teacher-forced token-level cross-entropy. Returns scalar Tensor (.aux = (sum_nll, n_tokens))."""
        cfg = self.cfg
        enc = self.encode(srcs, training)
        state = self.init_state(enc)
        y = ag.dropout(ag.embedding(self.emb_tgt, tgt_in), cfg.dropout, self.rng, training)
        ys = ag.time_slices(y)
        outs = []
        for t in range(tgt_in.shape[1]):
            o, state, _ = self.step(state, ys[t], enc, training)
            outs.append(o)
        o_all = ag.dropout(ag.stack_time(outs), cfg.dropout, self.rng, training)
        B, T = tgt_in.shape
        logits = ag.reshape(ag.linear(o_all, self.p["proj.W"], self.p["proj.b"]), (B * T, self.tgt_size))
        return ag.cross_entropy(logits, tgt_out.reshape(-1), tgt_mask.reshape(-1).astype(np.float64),
                                cfg.label_smoothing if training else 0.0)

    # ------------------------------------------------------------------ inference helpers (no graph)
    def next_logp(self, state, tokens: np.ndarray, enc):
        """tokens (B,) previous token ids -> (log-probs (B,V) float64, new_state, attention weights)."""
        with ag.no_grad():
            y = ag.embedding(self.emb_tgt, tokens)
            o, state, alphas = self.step(state, y, enc, training=False)
            logits = o.data @ self.p["proj.W"].data + self.p["proj.b"].data
        z = logits.astype(np.float64)
        z -= z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True)), state, alphas

    @staticmethod
    def expand_enc(enc, k):
        rep = lambda t: ag.const(np.repeat(t.data, k, axis=0))
        return {"outs": [rep(t) for t in enc["outs"]], "masks": [np.repeat(m, k, axis=0) for m in enc["masks"]],
                "kps": [rep(t) for t in enc["kps"]], "hc0": rep(enc["hc0"])}

    @staticmethod
    def head_enc(enc, n):
        h = lambda t: ag.const(t.data[:n])
        return {"outs": [h(t) for t in enc["outs"]], "masks": [m[:n] for m in enc["masks"]],
                "kps": [h(t) for t in enc["kps"]], "hc0": h(enc["hc0"])}

    @staticmethod
    def select_state(state, idx):
        sel = lambda t: None if t is None else ag.const(t.data[idx])
        return {"hc": sel(state["hc"]), "a": sel(state["a"])}
