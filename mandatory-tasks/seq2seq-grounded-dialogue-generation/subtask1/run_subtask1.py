#!/usr/bin/env python
"""Sub-task 1: basic LSTM encoder-decoder (no attention) for English -> target-language translation (French / Hindi / ...), trained from scratch.

    python run_subtask1.py                 # full run
    python run_subtask1.py --quick --toy   # 1-minute smoke test on a synthetic corpus
Outputs: results/ (report.md, metrics.json, no_attention.npz, vocab files, figures/)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from s2s import plots  # noqa: E402
from s2s.cli import add_translation_args, apply_quick, data_fingerprint  # noqa: E402
from s2s.models import ModelConfig  # noqa: E402
from s2s.pipeline_translation import (bleu_by_length, bleu_by_rare_words, evaluate_translation, md_table,  # noqa: E402
                                      prepare_translation, train_translation_model)
from s2s.data_translation import lang_name  # noqa: E402
from s2s.metrics import sentence_bleu  # noqa: E402
from s2s.tokenize import detokenize  # noqa: E402
from s2s.train import save_json  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = add_translation_args(argparse.ArgumentParser(), HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = apply_quick(ap.parse_args())
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    data = prepare_translation(args)
    st = data["stats"]
    print(f"[data] {data['source']}\n[data] train/val/test = {st['n_train']}/{st['n_val']}/{st['n_test']}  "
          f"src vocab {st['src_vocab']}  tgt vocab {st['tgt_vocab']}  test OOV src {st['src_oov_test']:.1%} tgt {st['tgt_oov_test']:.1%}")
    data["src_vocab"].save(os.path.join(args.out, "src_vocab.json"))
    data["tgt_vocab"].save(os.path.join(args.out, "tgt_vocab.json"))
    save_json({"fingerprint": data_fingerprint(args), "stats": st}, os.path.join(args.out, "meta.json"))

    cfg = ModelConfig(emb_dim=args.emb_dim, hid=args.hid, attn="none", dropout=args.dropout)
    model, hist = train_translation_model("no_attention", cfg, data, args, args.out)
    plots.plot_histories({"no attention": hist}, os.path.join(args.out, "figures", "loss_curves.png"))

    test = data["splits"]["test"]
    greedy, hyps = evaluate_translation(model, data, test, strategy="greedy", max_len=args.max_len + 10)
    val_greedy, _ = evaluate_translation(model, data, data["splits"]["val"], strategy="greedy", max_len=args.max_len + 10)
    by_len = bleu_by_length(test, hyps)
    by_rare = bleu_by_rare_words(test, hyps, data["src_vocab"], data["tgt_vocab"])
    plots.plot_bleu_by_length({"no attention": by_len}, os.path.join(args.out, "figures", "bleu_by_length.png"))
    print(f"[result] test BLEU (greedy) = {greedy['bleu']:.2f}   val BLEU = {val_greedy['bleu']:.2f}")
    for r in by_len:
        print(f"         length {r['bucket']:>6}: n={r['n']:4d}  BLEU {r['bleu']:.2f}")

    scored = sorted(range(len(test)), key=lambda i: sentence_bleu(test[i]["tgt_tok"], hyps[i]))
    long_idx = [i for i in range(len(test)) if len(test[i]["src_tok"]) >= 12][:4]
    ex_rows = [{"kind": k, "source": detokenize(test[i]["src_tok"]), "reference": detokenize(test[i]["tgt_tok"]),
                "model output": detokenize(hyps[i])} for k, idxs in
               (("typical", range(4)), ("long sentence", long_idx), ("worst", scored[:4])) for i in idxs]

    metrics = {"test": greedy, "val": val_greedy, "by_length": by_len, "by_rare_words": by_rare, "stats": st,
               "n_params": hist["n_params"], "best_val_nll": hist["best_val_loss"], "train_seconds": hist["train_seconds"],
               "config": vars(args)}
    save_json(metrics, os.path.join(args.out, "metrics.json"))

    short = [r for r in by_len if r["bucket"] in ("1-5", "6-10")]
    long_ = [r for r in by_len if r["bucket"] not in ("1-5", "6-10")]
    insight = ""
    if short and long_:
        sb, lb = np.mean([r["bleu"] for r in short]), np.mean([r["bleu"] for r in long_])
        insight = (f"BLEU is {sb:.1f} on sentences of up to 10 tokens but {lb:.1f} on longer ones "
                   f"({'a clear drop' if lb < 0.8 * sb else 'little difference'}), which is what a single fixed-size context "
                   "vector is expected to cause: the whole source has to be squeezed into one vector.")
    rare = ""
    if len(by_rare) == 2:
        a = {r["group"]: r["bleu"] for r in by_rare}
        rare = (f"Sentences containing an out-of-vocabulary word score {a['has <unk> word']:.1f} BLEU versus "
                f"{a['all words known']:.1f} for fully in-vocabulary sentences; rare words are replaced by `<unk>` and cannot be "
                "recovered by this model.")
    bullets = "\n".join(f"- {t}" for t in (insight, rare) if t) or "- (not enough sentences per bucket to draw a length / rare-word conclusion)"
    report = f"""# Sub-task 1 report: basic LSTM encoder-decoder (no attention)

{'> **WARNING: synthetic toy corpus - smoke test only. Do not report these numbers.**' + chr(10) if args.toy else ''}
## Approach
Encoder: embedding -> 1-layer LSTM; its final hidden/cell state is the single context vector. Decoder: embedding -> 1-layer LSTM initialised
with that context, trained with teacher forcing (cross-entropy, Adam, gradient clipping 5, dropout {args.dropout}, LR halved on plateau,
early stopping on validation loss). Everything - autograd, LSTM, loss, optimiser, decoding, BLEU - is implemented from scratch in NumPy.
Embedding {args.emb_dim}, hidden {args.hid}, {hist['n_params']:,} parameters, trained {hist['train_seconds']:.0f} s.

## Data
{data['source']}; language pair {lang_name(args.src_lang)} -> {lang_name(args.tgt_lang)}. Pairs are tokenised with a script-aware tokenizer (Devanagari matras and French accents/elisions are kept inside words), lower-cased, limited to {args.max_len} tokens, de-duplicated, and split
{st['n_train']}/{st['n_val']}/{st['n_test']} (train/val/test). Vocabularies are built on train only (min frequency {args.min_freq}, max {args.max_vocab}):
source {st['src_vocab']}, target {st['tgt_vocab']}. Test OOV rate: source {st['src_oov_test']:.1%}, target {st['tgt_oov_test']:.1%}.

## Results
Test BLEU (greedy decoding, corpus BLEU-4 with add-1 smoothing on n>1, on our tokenisation - not comparable to sacreBLEU numbers):
**{greedy['bleu']:.2f}** (validation {val_greedy['bleu']:.2f}); brevity penalty {greedy['bp']:.2f}.

![loss](figures/loss_curves.png)

## Where the basic model struggles
{md_table(by_len, ['bucket', 'n', 'bleu'])}

![bleu by length](figures/bleu_by_length.png)

{md_table(by_rare, ['group', 'n', 'bleu']) if by_rare else ''}

{bullets}

### Example outputs
{md_table(ex_rows, ['kind', 'source', 'reference', 'model output'])}

## Takeaways
The no-attention model gives the baseline for sub-task 2. Its failure modes (long sentences, rare words, repeated or truncated outputs) motivate
attention (a different context per output token) and better decoding.
"""
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[done] report -> {os.path.join(args.out, 'report.md')}")


if __name__ == "__main__":
    main()
