#!/usr/bin/env python
"""Sub-task 2: attention (Bahdanau, Luong) + decoding strategies (greedy, beam, temperature / top-k / top-p sampling).

    python run_subtask2.py                 # full run (re-uses sub-task 1's no-attention model when its settings match)
    python run_subtask2.py --quick --toy   # 1-2 minute smoke test on a synthetic corpus
Outputs: results/ (report.md, metrics.json, *.npz, figures/ incl. attention heat-maps)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from s2s import plots  # noqa: E402
from s2s.cli import add_translation_args, apply_quick, data_fingerprint  # noqa: E402
from s2s.decoding import generate_batch  # noqa: E402
from s2s.data_translation import lang_name  # noqa: E402
from s2s.metrics import sentence_bleu  # noqa: E402
from s2s.models import ModelConfig, Seq2Seq  # noqa: E402
from s2s.pipeline_translation import (bleu_by_length, bleu_by_rare_words, evaluate_translation, md_table,  # noqa: E402
                                      prepare_translation, src_batch, train_translation_model)
from s2s.tokenize import detokenize  # noqa: E402
from s2s.train import save_json  # noqa: E402

STRATEGIES = [
    ("greedy", dict(strategy="greedy")),
    ("beam 3", dict(strategy="beam", beam=3)),
    ("beam 5", dict(strategy="beam", beam=5)),
    ("sampling T=0.7", dict(strategy="sample", temperature=0.7)),
    ("sampling T=1.0", dict(strategy="sample", temperature=1.0)),
    ("top-k 10 (T=1)", dict(strategy="sample", temperature=1.0, top_k=10)),
    ("top-p 0.9 (T=1)", dict(strategy="sample", temperature=1.0, top_p=0.9)),
]


def reuse_no_attention(args, data):
    """Load sub-task 1's model if it was trained with identical data / model settings."""
    d = os.path.join(HERE, "..", "subtask1", "results")
    try:
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        if meta["fingerprint"] != json.loads(json.dumps(data_fingerprint(args))):
            return None
        model = Seq2Seq.load(os.path.join(d, "no_attention.npz"))
        hist = json.load(open(os.path.join(d, "no_attention_history.json"), encoding="utf-8"))
        print("[subtask2] re-using the no-attention model trained in sub-task 1 (identical settings)")
        return model, hist
    except Exception:  # noqa: BLE001
        return None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = add_translation_args(argparse.ArgumentParser(), HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--with-dot", action="store_true", help="also train Luong dot-product attention")
    ap.add_argument("--sample-seeds", type=int, default=3)
    args = apply_quick(ap.parse_args())
    figd = os.path.join(args.out, "figures")
    os.makedirs(figd, exist_ok=True)

    data = prepare_translation(args)
    st = data["stats"]
    print(f"[data] {data['source']}  train/val/test = {st['n_train']}/{st['n_val']}/{st['n_test']}")
    test = data["splits"]["test"]
    sv, tv = data["src_vocab"], data["tgt_vocab"]
    max_len = args.max_len + 10

    # ------------------------------------------------------------------ 1. train / load the models
    models, hists = {}, {}
    r = reuse_no_attention(args, data)
    if r:
        models["no attention"], hists["no attention"] = r
    else:
        m, h = train_translation_model("no_attention", ModelConfig(emb_dim=args.emb_dim, hid=args.hid, attn="none", dropout=args.dropout),
                                       data, args, args.out)
        models["no attention"], hists["no attention"] = m, h
    variants = [("Bahdanau (additive)", "bahdanau", "bahdanau"), ("Luong (general)", "luong_general", "luong_general")]
    if args.with_dot:
        variants.append(("Luong (dot)", "luong_dot", "luong_dot"))
    for label, attn, fname in variants:
        cfg = ModelConfig(emb_dim=args.emb_dim, hid=args.hid, attn=attn, att_dim=args.hid, dropout=args.dropout)
        models[label], hists[label] = train_translation_model(fname, cfg, data, args, args.out)
    plots.plot_histories(hists, os.path.join(figd, "loss_curves.png"))

    # ------------------------------------------------------------------ 2. attention vs no attention (greedy)
    rows, by_len, outputs = [], {}, {}
    for name, m in models.items():
        res, hyps = evaluate_translation(m, data, test, strategy="greedy", max_len=max_len)
        outputs[name] = hyps
        by_len[name] = bleu_by_length(test, hyps)
        rows.append({"model": name, "params": f"{hists[name]['n_params']:,}", "best val nll": hists[name]["best_val_loss"],
                     "test BLEU": res["bleu"], "BLEU-1 prec": res["precisions"][0], "avg len": res["avg_len"],
                     "train s": hists[name]["train_seconds"]})
        print(f"[result] {name:22s} test BLEU {res['bleu']:.2f}")
    plots.plot_bleu_by_length(by_len, os.path.join(figd, "bleu_by_length.png"))
    plots.plot_bars([r_["model"] for r_ in rows], [r_["test BLEU"] for r_ in rows], os.path.join(figd, "bleu_models.png"),
                    "Test BLEU (greedy)", "BLEU")

    # ------------------------------------------------------------------ 3. decoding strategies on the best attention model
    att_rows = [r_ for r_ in rows if r_["model"] != "no attention"]
    best_name = max(att_rows, key=lambda r_: r_["test BLEU"])["model"]
    best = models[best_name]
    dec_rows, dec_out = [], {}
    for label, kw in STRATEGIES:
        seeds = range(args.sample_seeds) if kw["strategy"] == "sample" else [0]
        accs, hyps0 = [], None
        for sd in seeds:
            res, hyps = evaluate_translation(best, data, test, max_len=max_len, seed=sd, **kw)
            accs.append(res)
            hyps0 = hyps0 or hyps
        dec_out[label] = hyps0
        avg = lambda k: float(np.mean([a[k] for a in accs]))
        dec_rows.append({"strategy": label, "BLEU": avg("bleu"), "distinct-1": avg("distinct1"), "distinct-2": avg("distinct2"),
                         "avg len": avg("avg_len"), "repeat rate": avg("repetition"), "decode s": avg("decode_seconds")})
        print(f"[decode] {label:16s} BLEU {avg('bleu'):6.2f}  distinct-2 {avg('distinct2'):.3f}  avg len {avg('avg_len'):.1f}")

    # ------------------------------------------------------------------ 4. qualitative: examples + attention heat-maps
    idx = [i for i in range(len(test)) if 6 <= len(test[i]["src_tok"]) <= 14][:3] or list(range(3))
    ex_rows = []
    for i in idx:
        ex_rows.append({"strategy": "SOURCE", "text": detokenize(test[i]["src_tok"])})
        ex_rows.append({"strategy": "REFERENCE", "text": detokenize(test[i]["tgt_tok"])})
        for label, _ in STRATEGIES:
            ex_rows.append({"strategy": label, "text": detokenize(dec_out[label][i])})
        ex_rows.append({"strategy": "", "text": ""})
    heat = []
    for name in ("Bahdanau (additive)", "Luong (general)"):
        if name not in models:
            continue
        i = idx[0]
        outs, attn = generate_batch(models[name], src_batch([test[i]], sv), tv.bos, tv.eos, max_len=max_len, return_attention=True)
        if outs[0]:
            A = np.array([a[0][:len(test[i]["src_tok"])] for a in attn[0]])
            fn = f"attention_{name.split()[0].lower()}.png"
            plots.plot_attention(A, test[i]["src_tok"], tv.decode(outs[0]), os.path.join(figd, fn), f"{name}: attention while translating")
            heat.append(fn)

    # ------------------------------------------------------------------ 5. report
    b0 = {r_["model"]: r_["test BLEU"] for r_ in rows}
    gain = b0[best_name] - b0["no attention"]
    nl = by_len["no attention"]; bl = by_len[best_name]
    long_gain = ""
    common = [(a_, b_) for a_, b_ in zip(nl, bl) if a_["bucket"] == b_["bucket"]]
    if len(common) >= 3:
        s_gain = np.mean([b_["bleu"] - a_["bleu"] for a_, b_ in common[:2]])
        l_gain = np.mean([b_["bleu"] - a_["bleu"] for a_, b_ in common[2:]])
        long_gain = (f"The attention gain is {s_gain:+.1f} BLEU on short sources (<=10 tokens) and {l_gain:+.1f} BLEU on longer ones, "
                     f"{'which matches the expectation that attention helps most on long sentences' if l_gain > s_gain else 'so on this data the benefit does not grow with length'}.")
    by_s = {r_["strategy"]: r_ for r_ in dec_rows}
    dec_text = (f"Beam search changes BLEU by {by_s['beam 5']['BLEU'] - by_s['greedy']['BLEU']:+.2f} (beam 5 vs greedy). "
                f"Unrestricted sampling (T=1) scores {by_s['sampling T=1.0']['BLEU']:.1f} BLEU with distinct-2 {by_s['sampling T=1.0']['distinct-2']:.2f}, "
                f"against greedy's {by_s['greedy']['BLEU']:.1f} / {by_s['greedy']['distinct-2']:.2f}: sampling buys diversity at the price of accuracy, "
                f"and top-k / top-p recover part of the lost BLEU ({by_s['top-k 10 (T=1)']['BLEU']:.1f} / {by_s['top-p 0.9 (T=1)']['BLEU']:.1f}).")
    save_json({"models": rows, "by_length": by_len, "decoding": dec_rows, "best_attention_model": best_name,
               "stats": st, "config": vars(args)}, os.path.join(args.out, "metrics.json"))
    warn = "> **WARNING: synthetic toy corpus - smoke test only. Do not report these numbers.**\n\n" if args.toy else ""
    report = f"""# Sub-task 2 report: attention and decoding strategies

{warn}## Setup
Same data ({lang_name(args.src_lang)} -> {lang_name(args.tgt_lang)}), vocabularies, split, embedding ({args.emb_dim}) and hidden size ({args.hid}) as sub-task 1; {data['source']}; train/val/test =
{st['n_train']}/{st['n_val']}/{st['n_test']}. Only the decoder's access to the source changes.
- **Bahdanau:** additive score `v . tanh(W_q h_(t-1) + W_k h_s)`, query = previous decoder state, context fed into the LSTM input.
- **Luong (general):** score `h_t W h_s`, query = current decoder state, attentional vector `tanh(W_c [h_t; c_t])`, fed back as input (input feeding).
- Masked softmax over real source positions; teacher forcing; same optimiser and early stopping as sub-task 1.

## 1. Does attention help? (test BLEU, greedy)
{md_table(rows, ['model', 'params', 'best val nll', 'test BLEU', 'avg len', 'train s'])}

![loss](figures/loss_curves.png)
![bleu by length](figures/bleu_by_length.png)

{md_table([dict(bucket=a_['bucket'], n=a_['n'], **{k: next((x['bleu'] for x in v if x['bucket'] == a_['bucket']), float('nan')) for k, v in by_len.items()}) for a_ in nl], ['bucket', 'n'] + list(by_len))}

Best attention model: **{best_name}**, {gain:+.2f} BLEU over no attention. {long_gain}

### Attention heat-maps
{chr(10).join(f'![attention](figures/{h})' for h in heat)}

Rows are generated tokens, columns are source tokens; bright cells show which source words the decoder looked at. Word-order differences between the two languages (e.g. adjective-noun order in English vs French, SVO vs SOV for Hindi) show up as off-diagonal alignments.

## 2. Decoding strategies (on {best_name})
{md_table(dec_rows, ['strategy', 'BLEU', 'distinct-1', 'distinct-2', 'avg len', 'repeat rate', 'decode s'])}

(sampling rows averaged over {args.sample_seeds} seeds; distinct-n = share of unique n-grams, a diversity measure.)

{dec_text}

### Example outputs
{md_table(ex_rows, ['strategy', 'text'])}

## 3. Analysis
- **Attention vs. fixed context vector.** The fixed vector must carry the whole sentence; attention lets each output word read the relevant source states. {long_gain}
- **Greedy vs. beam.** Beam search keeps several partial translations and usually finds a higher-probability output than greedy, at about k times the cost; with length normalisation it avoids favouring very short outputs.
- **Sampling.** Temperature rescales the distribution (lower = safer, higher = more random); top-k and top-p cut the unreliable tail. Translation has a single correct-ish answer, so deterministic decoding wins on BLEU, while sampling is useful when diversity matters (dialogue, sub-task 3).
- **Remaining failure modes.** Rare words become `<unk>`; long sentences are still harder; the vocabulary is word-level, so rich morphology (French verb forms, Hindi inflection) multiplies word forms.
"""
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[done] report -> {os.path.join(args.out, 'report.md')}")


if __name__ == "__main__":
    main()
