import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from s2s import autograd as ag  # noqa: E402

ag.DTYPE = np.float64                        # exact finite-difference checks (set BEFORE building models)

from s2s.decoding import beam_search, generate_batch  # noqa: E402
from s2s.models import ModelConfig, Seq2Seq  # noqa: E402
from s2s.tokenize import pad_batch  # noqa: E402

V = 12


def _batch(rng, n_src, B=3):
    srcs = []
    for _ in range(n_src):
        seqs = [list(rng.integers(4, V, rng.integers(2, 6))) for _ in range(B)]
        ids, _, mask = pad_batch(seqs, 0)
        srcs.append((ids, mask.astype(np.float64)))
    tg = [list(rng.integers(4, V, rng.integers(2, 5))) for _ in range(B)]
    tin, _, m = pad_batch([[2] + s for s in tg], 0)
    tout, _, _ = pad_batch([s + [3] for s in tg], 0)
    return srcs, tin, tout, m.astype(np.float64)


def test_full_model_gradients_all_variants():
    rng = np.random.default_rng(1)
    for attn in ("none", "bahdanau", "luong_general", "luong_dot"):
        for ns in (1, 2):
            m = Seq2Seq(ModelConfig(emb_dim=5, hid=6, attn=attn, n_sources=ns, att_dim=4, dropout=0.0),
                        [V] * ns, V, seed=3)
            srcs, tin, tout, tm = _batch(rng, ns)
            def f():
                return m.loss(srcs, tin, tout, tm, training=False)
            m.zero_grad(); f().backward()
            worst = 0.0
            for name, p in m.p.items():
                g = np.zeros_like(p.data) if p.grad is None else p.grad
                flat, gf = p.data.reshape(-1), g.reshape(-1)
                for k in rng.choice(flat.size, size=min(6, flat.size), replace=False):
                    old = flat[k]
                    flat[k] = old + 1e-6; lp = float(f().data)
                    flat[k] = old - 1e-6; lm = float(f().data)
                    flat[k] = old
                    num = (lp - lm) / 2e-6
                    worst = max(worst, abs(num - gf[k]) / max(1e-4, abs(num) + abs(gf[k])))
            assert worst < 1e-5, f"{attn}/{ns} sources: gradient error {worst:.2e}"


def test_padding_does_not_change_the_loss():
    rng = np.random.default_rng(2)
    m = Seq2Seq(ModelConfig(emb_dim=5, hid=6, attn="bahdanau", att_dim=4, dropout=0.0), [V], V, seed=0)
    srcs, tin, tout, tm = _batch(rng, 1, B=1)
    a = float(m.loss(srcs, tin, tout, tm, training=False).data)
    ids, mask = srcs[0]
    pad_ids = np.concatenate([ids, np.zeros((1, 4), dtype=ids.dtype)], axis=1)
    pad_mask = np.concatenate([mask, np.zeros((1, 4))], axis=1)
    tin2 = np.concatenate([tin, np.zeros((1, 3), dtype=tin.dtype)], axis=1)
    tout2 = np.concatenate([tout, np.zeros((1, 3), dtype=tout.dtype)], axis=1)
    tm2 = np.concatenate([tm, np.zeros((1, 3))], axis=1)
    b = float(m.loss([(pad_ids, pad_mask)], tin2, tout2, tm2, training=False).data)
    assert abs(a - b) < 1e-9, (a, b)


def test_decoding_functions_run_and_agree_on_degenerate_cases():
    rng = np.random.default_rng(3)
    for attn in ("none", "bahdanau", "luong_general"):
        m = Seq2Seq(ModelConfig(emb_dim=5, hid=6, attn=attn, att_dim=4, dropout=0.0), [V], V, seed=1)
        srcs, *_ = _batch(rng, 1, B=1)
        g = generate_batch(m, srcs, 2, 3, max_len=8)[0]
        b1 = beam_search(m, srcs, 2, 3, beam=1, max_len=8, length_penalty=0.0)
        assert g == b1, (attn, g, b1)                           # beam=1 without length penalty == greedy
        b4 = beam_search(m, srcs, 2, 3, beam=4, max_len=8)
        s = generate_batch(m, srcs, 2, 3, max_len=8, mode="sample", temperature=0.8, top_k=5, top_p=0.9,
                           rng=np.random.default_rng(0))[0]
        assert all(0 <= t < V for t in g + b4 + s) and len(g) <= 8 and len(b4) <= 8


def test_beam_score_is_not_worse_than_greedy():
    rng = np.random.default_rng(4)
    m = Seq2Seq(ModelConfig(emb_dim=5, hid=6, attn="luong_dot", dropout=0.0), [V], V, seed=2)
    srcs, *_ = _batch(rng, 1, B=1)

    def seq_logp(toks):
        enc = m.encode(srcs, False); st = m.init_state(enc); prev = np.array([2]); tot = 0.0
        for t in toks + [3]:
            lp, st, _ = m.next_logp(st, prev, enc); tot += lp[0, t]; prev = np.array([t])
        return tot
    g = generate_batch(m, srcs, 2, 3, max_len=6)[0]
    b = beam_search(m, srcs, 2, 3, beam=6, max_len=6, length_penalty=0.0)
    if len(g) < 6:                                              # greedy finished with eos inside the limit
        assert seq_logp(b) >= seq_logp(g) - 1e-9


def test_save_load_roundtrip(tmp_path=None):
    import tempfile
    m = Seq2Seq(ModelConfig(emb_dim=5, hid=6, attn="bahdanau", n_sources=2, att_dim=4), [V, V], V, seed=0)
    path = os.path.join(tempfile.mkdtemp(), "m.npz")
    m.save(path)
    m2 = Seq2Seq.load(path)
    assert all(np.array_equal(m.p[k].data, m2.p[k].data) for k in m.p)

def test_ban_ids_never_generated():
    """Decoding with ban_ids must never output a banned token (used to stop <unk>-only replies in sub-task 3)."""
    import numpy as np
    from s2s.models import ModelConfig, Seq2Seq
    from s2s.decoding import beam_search, generate_batch
    m = Seq2Seq(ModelConfig(emb_dim=8, hid=8, attn="bahdanau", att_dim=8, dropout=0.0), [12], 12, seed=1)
    ids = np.array([[4, 5, 6, 7]]); mask = np.ones((1, 4), dtype=np.float32)
    for mode in ("greedy", "sample"):
        out = generate_batch(m, [(ids, mask)], 2, 3, max_len=15, mode=mode, ban_ids=(1, 5, 6, 7, 8))[0]
        assert not set(out) & {1, 5, 6, 7, 8}, (mode, out)
    out = beam_search(m, [(ids, mask)], 2, 3, beam=3, max_len=15, ban_ids=(1, 5, 6, 7, 8))
    assert not set(out) & {1, 5, 6, 7, 8}, out

if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("PASS", k)
