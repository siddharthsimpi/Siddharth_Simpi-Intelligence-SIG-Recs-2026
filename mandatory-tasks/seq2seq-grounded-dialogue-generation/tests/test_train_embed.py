import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from s2s import autograd as ag  # noqa: E402
from s2s.embeddings import train_sgns  # noqa: E402
from s2s.models import ModelConfig, Seq2Seq  # noqa: E402
from s2s.train import collate, eval_loss, iterate_batches, train  # noqa: E402


def test_copy_task_is_learned_by_every_variant():
    """Sanity: with enough capacity each variant must learn to copy a short random sequence (loss << ln V)."""
    rng = np.random.default_rng(0)
    V = 14
    data = []
    for _ in range(300):
        s = [int(x) for x in rng.integers(4, V, rng.integers(3, 6))]
        data.append({"srcs": [s], "tgt": s})
    col = lambda b: collate(b, 0, 2, 3, 1)
    for attn in ("none", "bahdanau", "luong_general"):
        m = Seq2Seq(ModelConfig(emb_dim=24, hid=48, attn=attn, att_dim=32, dropout=0.0), [V], V, seed=0)
        h = train(m, data[:250], data[250:], col, epochs=14, batch_size=32, lr=5e-3, patience=20, log=lambda *_: None)
        vl, _ = eval_loss(m, data[250:], col)
        assert vl < 1.0, (attn, vl)                      # ln(14)=2.64 at chance
        assert h["train_loss"][-1] < h["train_loss"][0]


def test_batching_covers_every_example_once():
    ex = [{"srcs": [[1] * (i % 7 + 1)], "tgt": [1] * (i % 5 + 1)} for i in range(103)]
    seen = [id(e) for b in iterate_batches(ex, 16, np.random.default_rng(0)) for e in b]
    assert sorted(seen) == sorted(id(e) for e in ex)


def test_sgns_groups_cooccurring_words():
    rng = np.random.default_rng(0)
    A, B = list(range(4, 14)), list(range(14, 24))                 # two "topics"
    sents = [list(rng.choice(A if rng.random() < 0.5 else B, 8)) for _ in range(1500)]
    emb = train_sgns(sents, 24, dim=16, window=3, negatives=4, epochs=6, subsample=1.0, log=lambda *_: None)
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    within = np.mean([e[i] @ e[j] for i in A for j in A if i != j])
    across = np.mean([e[i] @ e[j] for i in A for j in B])
    assert within > across + 0.2, (within, across)


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("PASS", k)
