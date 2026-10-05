"""Training, decoding and evaluation helpers for sub-task 3 (history + document -> reply)."""
from __future__ import annotations

import os
import time

import numpy as np

from . import metrics
from .decoding import beam_search, generate_batch
from .models import ModelConfig, Seq2Seq
from .tokenize import pad_batch
from .train import collate, save_json, train


def make_model(args, vocab_size, n_sources, seed=0, emb_matrix=None):
    cfg = ModelConfig(emb_dim=args.emb_dim, hid=args.hid, attn=args.attn, n_sources=n_sources, dropout=args.dropout,
                      att_dim=args.hid, share_embeddings=True)
    m = Seq2Seq(cfg, [vocab_size] * n_sources, vocab_size, seed=seed)
    if emb_matrix is not None:
        m.set_embeddings({"all": emb_matrix}, freeze=getattr(args, "freeze_emb", False))
    return m


def train_dialogue_model(name, model, data, args, out_dir, log=print):
    v, n_src = data["vocab"], model.cfg.n_sources
    col = lambda b: collate(b, v.pad, v.bos, v.eos, n_src)
    log(f"[{name}] parameters: {model.n_params():,}")
    hist = train(model, data["examples"]["train"], data["examples"]["val"], col, epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, patience=args.patience, seed=args.seed, log=log,
                 max_seconds=args.max_seconds, ckpt_path=os.path.join(out_dir, f"{name}.npz"), tag=name)
    hist["n_params"] = model.n_params()
    save_json(hist, os.path.join(out_dir, f"{name}_history.json"))
    return hist


def src_batch(examples, vocab, n_sources, doc_override=None):
    srcs = []
    for k in range(n_sources):
        seqs = [(doc_override[i] if (k == 1 and doc_override is not None) else e["srcs"][k]) for i, e in enumerate(examples)]
        ids, _, mask = pad_batch(seqs, vocab.pad)
        srcs.append((ids, mask))
    return srcs


def generate(model, vocab, examples, strategy="greedy", beam=3, max_len=30, no_repeat_ngram=3, temperature=1.0,
             top_k=0, top_p=1.0, batch_size=64, seed=0, doc_override=None, min_len=0, allow_unk=False):
    n_src = model.cfg.n_sources
    ban = (vocab.pad,) if allow_unk else (vocab.pad, vocab.unk)     # a reply made of <unk> tokens is useless to read
    rng = np.random.default_rng(seed)
    hyps = []
    if strategy == "beam":
        for i, e in enumerate(examples):
            ov = None if doc_override is None else [doc_override[i]]
            ids = beam_search(model, src_batch([e], vocab, n_src, ov), vocab.bos, vocab.eos, beam=beam, max_len=max_len,
                              no_repeat_ngram=no_repeat_ngram, min_len=min_len, ban_ids=ban)
            hyps.append(vocab.decode(ids))
        return hyps
    for s in range(0, len(examples), batch_size):
        chunk = examples[s:s + batch_size]
        ov = None if doc_override is None else doc_override[s:s + batch_size]
        outs = generate_batch(model, src_batch(chunk, vocab, n_src, ov), vocab.bos, vocab.eos, max_len=max_len,
                              mode="greedy" if strategy == "greedy" else "sample", temperature=temperature, top_k=top_k,
                              top_p=top_p, no_repeat_ngram=no_repeat_ngram, rng=rng, min_len=min_len, ban_ids=ban)
        hyps += [vocab.decode(o) for o in outs]
    return hyps


def score(examples, hyps):
    refs = [e["resp_tok"] for e in examples]
    out = metrics.evaluate_generation(refs, hyps, [e["doc_tok"] for e in examples])
    docs = [e["doc_tok"] for e in examples]
    hits = []
    for h, d in zip(hyps, docs):
        ds = set(d)
        content = metrics.content_words(h)
        hits.append(any(t in ds for t in content) if content else False)
    out["doc_word_hit"] = float(np.mean(hits))
    top5 = [c for _, c in __import__("collections").Counter(" ".join(h) for h in hyps).most_common(5)]
    out["top5_reply_share"] = sum(top5) / max(len(hyps), 1)           # share of outputs that are one of the 5 most common replies
    out["empty_rate"] = float(np.mean([len(h) == 0 for h in hyps]))
    return out


def score_by_bucket(examples, hyps, bucket_fn):
    groups = {}
    for i, e in enumerate(examples):
        groups.setdefault(bucket_fn(e["share"]), []).append(i)
    rows = []
    for b, idx in groups.items():
        if len(idx) < 5:
            continue
        refs, hs = [examples[i]["resp_tok"] for i in idx], [hyps[i] for i in idx]
        r = metrics.rouge_scores(refs, hs)
        rows.append({"bucket": b, "n": len(idx), "BLEU-1": metrics.corpus_bleu(refs, hs, max_n=1)["bleu"],
                     "BLEU-2": metrics.corpus_bleu(refs, hs, max_n=2)["bleu"], "ROUGE-L": r["rougeL"],
                     "avg ref len": float(np.mean([len(x) for x in refs])), "avg gen len": float(np.mean([len(x) for x in hs]))})
    order = {"mostly English": 0, "mixed": 1, "mostly Hindi": 2, "unknown": 3}
    return sorted(rows, key=lambda r: order.get(r["bucket"], 9))
