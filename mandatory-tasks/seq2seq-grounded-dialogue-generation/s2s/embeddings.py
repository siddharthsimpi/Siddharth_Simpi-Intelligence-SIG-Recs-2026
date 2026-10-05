"""Word embeddings trained from scratch: skip-gram with negative sampling (Mikolov et al. 2013), vectorised NumPy.

For code-mixed Hinglish this is what 'generate embeddings from scratch' means here: the vectors are learned only from the
task's own text (dialogue turns + grounding documents), so romanised Hindi words, English words and spelling variants
all get vectors from their actual co-occurrence statistics. No pretrained vectors are used anywhere.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _apply(W, rows, grads, lr):
    """Row-averaged mini-batch SGD step: gradients of the same row are averaged (not summed), which keeps very frequent
    words stable when thousands of pairs share one batch (sequential SGD would see them one at a time)."""
    uniq, inv = np.unique(rows, return_inverse=True)
    acc = np.zeros((len(uniq), W.shape[1]))
    np.add.at(acc, inv, grads)
    W[uniq] -= lr * acc / np.bincount(inv)[:, None]


def train_sgns(sentences, vocab_size, dim=100, window=4, negatives=5, epochs=5, lr=0.5, batch=1024,
               subsample=1e-3, min_count=1, seed=0, log=print, max_pairs_per_epoch=3_000_000, skip_ids=(0, 1, 2, 3)):
    """sentences: list of id lists. Returns the input-embedding matrix (vocab_size, dim) as float32.
    Frequent words are sub-sampled; negatives are drawn from the unigram^0.75 distribution."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(vocab_size)
    for s in sentences:
        for t in s:
            counts[t] += 1
    counts_smooth = counts ** 0.75
    for i in skip_ids:
        counts_smooth[i] = 0
    noise = counts_smooth / counts_smooth.sum()
    freq = counts / max(counts.sum(), 1)
    keep_p = np.minimum(1.0, np.sqrt(subsample / np.maximum(freq, 1e-12)) + subsample / np.maximum(freq, 1e-12))
    W_in = ((rng.random((vocab_size, dim)) - 0.5) / dim).astype(np.float64)
    W_out = np.zeros((vocab_size, dim))
    flat = [np.array(s, dtype=np.int64) for s in sentences if len(s) > 1]
    total_steps = None
    step = 0
    for ep in range(epochs):
        centers, contexts = [], []
        for s in flat:
            s = s[rng.random(len(s)) < keep_p[s]]
            n = len(s)
            if n < 2:
                continue
            w = rng.integers(1, window + 1, size=n)                     # dynamic window
            for off in range(1, window + 1):
                sel = np.where(w >= off)[0]
                a, b = sel[sel + off < n], None
                if len(a):
                    centers += [s[a], s[a + off]]
                    contexts += [s[a + off], s[a]]
        if not centers:
            break
        c, o = np.concatenate(centers), np.concatenate(contexts)
        perm = rng.permutation(len(c))[:max_pairs_per_epoch]
        c, o = c[perm], o[perm]
        n_batches = (len(c) + batch - 1) // batch
        if total_steps is None:
            total_steps = n_batches * epochs
        loss_sum = 0.0
        for bi in range(n_batches):
            cb, ob = c[bi * batch:(bi + 1) * batch], o[bi * batch:(bi + 1) * batch]
            neg = rng.choice(vocab_size, size=(len(cb), negatives), p=noise)
            u = W_in[cb]                                                   # (B,d)
            vp = W_out[ob]                                                 # (B,d)
            vn = W_out[neg]                                                # (B,k,d)
            sp = _sigmoid((u * vp).sum(1))                                 # (B,)
            sn = _sigmoid(np.einsum("bd,bkd->bk", u, vn))                  # (B,k)
            loss_sum += float(-np.log(sp + 1e-10).sum() - np.log(1 - sn + 1e-10).sum())
            gu = (sp - 1)[:, None] * vp + np.einsum("bk,bkd->bd", sn, vn)
            gvp = (sp - 1)[:, None] * u
            gvn = sn[:, :, None] * u[:, None, :]
            cur_lr = lr * max(0.05, 1 - step / max(total_steps, 1))
            _apply(W_in, cb, gu, cur_lr)
            _apply(W_out, np.concatenate([ob, neg.reshape(-1)]), np.concatenate([gvp, gvn.reshape(-1, dim)]), cur_lr)
            step += 1
        log(f"[sgns] epoch {ep + 1}/{epochs}  pairs {len(c):,}  mean loss {loss_sum / len(c):.4f}")
    emb = W_in.astype(np.float32)
    emb[list(skip_ids)] = 0.0
    seen = counts >= max(min_count, 1)
    rng2 = np.random.default_rng(seed + 1)
    emb[~seen] = rng2.normal(0, 0.05, (int((~seen).sum()), dim))          # unseen rows: small random vectors
    return emb


def nearest(emb, vocab, word, k=5):
    """Cosine nearest neighbours of `word` (for qualitative inspection)."""
    if word not in vocab.stoi:
        return []
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = e @ e[vocab.stoi[word]]
    sims[[vocab.pad, vocab.unk, vocab.bos, vocab.eos]] = -1
    sims[vocab.stoi[word]] = -1
    return [(vocab.itos[i], float(sims[i])) for i in np.argsort(-sims)[:k]]
