"""Shared training / decoding / evaluation helpers for sub-tasks 1 and 2 (single-source translation)."""
from __future__ import annotations

import os
import time

import numpy as np

from . import metrics
from .decoding import beam_search, generate_batch
from .models import ModelConfig, Seq2Seq
from .data_translation import (build_translation_data, ensure_translation_data, infer_langs, lang_name,
                               make_toy_en_hi)
from .tokenize import pad_batch
from .train import collate, save_json, train

LENGTH_EDGES = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 999)]


def collate_fn(data):
    sv, tv = data["src_vocab"], data["tgt_vocab"]
    return lambda b: collate(b, tv.pad, tv.bos, tv.eos, 1)


def train_translation_model(name, cfg: ModelConfig, data, args, out_dir, log=print):
    sv, tv = data["src_vocab"], data["tgt_vocab"]
    model = Seq2Seq(cfg, [len(sv)], len(tv), seed=args.seed)
    log(f"[{name}] parameters: {model.n_params():,}")
    ckpt = os.path.join(out_dir, f"{name}.npz")
    hist = train(model, data["splits"]["train"], data["splits"]["val"], collate_fn(data), epochs=args.epochs,
                 batch_size=args.batch_size, lr=args.lr, patience=args.patience, seed=args.seed, log=log,
                 max_seconds=args.max_seconds, ckpt_path=ckpt, tag=name)
    hist["n_params"] = model.n_params()
    save_json(hist, os.path.join(out_dir, f"{name}_history.json"))
    return model, hist


def src_batch(examples, sv):
    ids, _, mask = pad_batch([e["srcs"][0] for e in examples], sv.pad)
    return [(ids, mask)]


def translate(model, data, examples, strategy="greedy", beam=5, temperature=1.0, top_k=0, top_p=1.0, max_len=40,
              batch_size=64, seed=0, no_repeat_ngram=0, length_penalty=0.6):
    """Decode every example. strategy: greedy | beam | sample (temperature / top_k / top_p). Returns token lists."""
    sv, tv = data["src_vocab"], data["tgt_vocab"]
    rng = np.random.default_rng(seed)
    hyps = []
    if strategy == "beam":
        for e in examples:
            ids = beam_search(model, src_batch([e], sv), tv.bos, tv.eos, beam=beam, max_len=max_len,
                              length_penalty=length_penalty, no_repeat_ngram=no_repeat_ngram)
            hyps.append(tv.decode(ids))
        return hyps
    for s in range(0, len(examples), batch_size):
        chunk = examples[s:s + batch_size]
        outs = generate_batch(model, src_batch(chunk, sv), tv.bos, tv.eos, max_len=max_len,
                              mode="greedy" if strategy == "greedy" else "sample", temperature=temperature,
                              top_k=top_k, top_p=top_p, no_repeat_ngram=no_repeat_ngram, rng=rng)
        hyps += [tv.decode(o) for o in outs]
    return hyps


def evaluate_translation(model, data, examples, **decode_kw):
    t0 = time.time()
    hyps = translate(model, data, examples, **decode_kw)
    refs = [e["tgt_tok"] for e in examples]
    out = metrics.corpus_bleu(refs, hyps)
    out.update({"distinct1": metrics.distinct_n(hyps, 1), "distinct2": metrics.distinct_n(hyps, 2),
                "avg_len": float(np.mean([len(h) for h in hyps])), "repetition": metrics.repetition_rate(hyps),
                "decode_seconds": time.time() - t0})
    return out, hyps


def bleu_by_length(examples, hyps, edges=LENGTH_EDGES):
    rows = []
    for lo, hi in edges:
        idx = [i for i, e in enumerate(examples) if lo <= len(e["src_tok"]) <= hi]
        if len(idx) >= 5:
            rows.append({"bucket": f"{lo}-{hi}" if hi < 999 else f"{lo}+", "n": len(idx),
                         "bleu": metrics.corpus_bleu([examples[i]["tgt_tok"] for i in idx], [hyps[i] for i in idx])["bleu"]})
    return rows


def bleu_by_rare_words(examples, hyps, src_vocab, tgt_vocab):
    """BLEU on sentences whose source (or reference) contains out-of-vocabulary words vs. fully in-vocabulary ones."""
    groups = {"all words known": [], "has <unk> word": []}
    for i, e in enumerate(examples):
        has_unk = any(t not in src_vocab.stoi for t in e["src_tok"]) or any(t not in tgt_vocab.stoi for t in e["tgt_tok"])
        groups["has <unk> word" if has_unk else "all words known"].append(i)
    rows = []
    for k, idx in groups.items():
        if len(idx) >= 5:
            rows.append({"group": k, "n": len(idx),
                         "bleu": metrics.corpus_bleu([examples[i]["tgt_tok"] for i in idx], [hyps[i] for i in idx])["bleu"]})
    return rows


def prepare_translation(args):
    """Load the corpus selected by the CLI args and build vocabularies / encoded splits."""
    if args.toy:
        args.src_lang, args.tgt_lang = "en", "hi"
        pairs, source = make_toy_en_hi(3000, args.seed), "SYNTHETIC toy corpus (smoke test only)"
    else:
        folder = args.data or os.path.join(args.data_dir, "translation")
        if (args.src_lang is None or args.tgt_lang is None) and os.path.exists(folder):
            src, tgt = infer_langs(folder)
            args.src_lang, args.tgt_lang = args.src_lang or src, args.tgt_lang or tgt
        args.src_lang, args.tgt_lang = args.src_lang or "en", args.tgt_lang or "hi"
        print(f"[data] language pair: {lang_name(args.src_lang)} -> {lang_name(args.tgt_lang)} "
              f"(override with --src-lang / --tgt-lang)")
        pairs, source = ensure_translation_data(args.data, args.data_dir, args.src_lang, args.tgt_lang,
                                                args.src_col, args.tgt_col)
    data = build_translation_data(pairs, args.max_len, args.min_freq, args.max_vocab, args.seed, args.max_train,
                                  max_eval=args.max_eval)
    data["source"] = source
    return data


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(out)
