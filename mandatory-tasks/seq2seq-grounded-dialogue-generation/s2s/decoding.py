"""Decoding strategies: greedy, beam search (length-normalised), temperature / top-k / top-p (nucleus) sampling.

All strategies work on any model exposing encode / init_state / next_logp (see models.Seq2Seq), so the same code decodes
translation and grounded dialogue. `no_repeat_ngram=n` forbids repeating any n-gram within one hypothesis.
"""
from __future__ import annotations

import numpy as np

from . import autograd as ag


def _banned(prefix, n):
    """Token ids that would complete an n-gram already present in `prefix` (list of ids)."""
    if n <= 0 or len(prefix) < n:
        return ()
    head = tuple(prefix[-(n - 1):]) if n > 1 else ()
    return {prefix[i + n - 1] for i in range(len(prefix) - n + 1) if tuple(prefix[i:i + n - 1]) == head}


def generate_batch(model, srcs, bos, eos, max_len=40, mode="greedy", temperature=1.0, top_k=0, top_p=1.0,
                                      no_repeat_ngram=0, rng=None, min_len=0, return_attention=False, ban_ids=()):
    """Batched greedy / sampling decoding. srcs = [(ids, mask), ...]. Returns list of id lists (without eos)."""
    rng = rng or np.random.default_rng(0)
    with ag.no_grad():
        enc = model.encode(srcs, training=False)
        state = model.init_state(enc)
    B = srcs[0][0].shape[0]
    tokens = np.full(B, bos, dtype=np.int64)
    done = np.zeros(B, bool)
    out = [[] for _ in range(B)]
    attn = [[] for _ in range(B)] if return_attention else None
    for t in range(max_len):
        logp, state, alphas = model.next_logp(state, tokens, enc)
        logp[:, bos] = -np.inf
        for bid in ban_ids:
            logp[:, bid] = -np.inf
        if t < min_len:
            logp[:, eos] = -np.inf
        if no_repeat_ngram:
            for b in range(B):
                for tok in _banned(out[b], no_repeat_ngram):
                    logp[b, tok] = -np.inf
        if mode == "greedy":
            nxt = logp.argmax(axis=1)
        else:
            nxt = np.array([_sample_one(logp[b], temperature, top_k, top_p, rng) for b in range(B)])
        for b in range(B):
            if done[b]:
                continue
            if nxt[b] == eos:
                done[b] = True
            else:
                out[b].append(int(nxt[b]))
                if return_attention:
                    attn[b].append([a[b].copy() for a in alphas])
        tokens = np.where(done, eos, nxt)
        if done.all():
            break
    return (out, attn) if return_attention else out


def _sample_one(logp, temperature, top_k, top_p, rng):
    z = logp / max(temperature, 1e-6)
    z = z - z.max()
    p = np.exp(z)
    if top_k and top_k < len(p):
        cut = np.partition(p, -top_k)[-top_k]
        p = np.where(p >= cut, p, 0.0)
    if top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order]) / p.sum()
        keep = order[: int(np.searchsorted(cum, top_p) + 1)]
        mask = np.zeros_like(p)
        mask[keep] = 1.0
        p = p * mask
    p = p / p.sum()
    return int(rng.choice(len(p), p=p))


def beam_search(model, srcs, bos, eos, beam=5, max_len=40, length_penalty=0.6, no_repeat_ngram=0, min_len=0, ban_ids=()):
    """Beam search for ONE example (srcs batch size 1). Score = sum log p / ((5+len)/6)^alpha (GNMT length penalty)."""
    with ag.no_grad():
        enc1 = model.encode(srcs, training=False)
        enc_full = model.expand_enc(enc1, beam)
        state = model.init_state(enc1)
    alive = [([], 0.0)]                         # (token ids, cumulative log prob)
    finished = []                               # (token ids, normalised score)

    def lp(n):
        return ((5.0 + n) / 6.0) ** length_penalty

    prev = np.array([bos], dtype=np.int64)
    for t in range(max_len):
        n = len(alive)
        enc_n = enc_full if n == beam else model.head_enc(enc_full, n)
        logp, state, _ = model.next_logp(state, prev, enc_n)
        logp[:, bos] = -np.inf
        for bid in ban_ids:
            logp[:, bid] = -np.inf
        if t < min_len:
            logp[:, eos] = -np.inf
        for i, (toks, _) in enumerate(alive):
            for tok in _banned(toks, no_repeat_ngram):
                logp[i, tok] = -np.inf
        total = np.array([s for _, s in alive])[:, None] + logp
        flat = total.reshape(-1)
        top = np.argpartition(-flat, min(2 * beam, len(flat) - 1))[: 2 * beam]
        top = top[np.argsort(-flat[top])]
        new_alive, parents = [], []
        for rank, idx in enumerate(top):
            i, tok = divmod(int(idx), logp.shape[1])
            score = float(flat[idx])
            if not np.isfinite(score):
                continue
            if tok == eos:
                if rank < beam:                  # an <eos> only closes a hypothesis if it is among the top `beam` candidates
                    finished.append((alive[i][0], score / lp(len(alive[i][0]) + 1)))
            elif len(new_alive) < beam:
                new_alive.append((alive[i][0] + [tok], score))
                parents.append(i)
        if not new_alive:
            break
        best_alive = max(s / lp(len(tk) + 1) for tk, s in new_alive)
        if len(finished) >= beam and max(s for _, s in finished) >= best_alive:
            break
        state = model.select_state(state, np.array(parents))
        prev = np.array([tk[-1] for tk, _ in new_alive], dtype=np.int64)
        alive = new_alive
    if not finished:
        finished = [(tk, s / lp(len(tk))) for tk, s in alive]
    return max(finished, key=lambda x: x[1])[0]
