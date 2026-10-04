#!/usr/bin/env python
"""Sub-task 3: document-grounded Hinglish dialogue generation (CMU Hinglish DoG), built from scratch.

    python inspect_dataset.py                 # look at the data first
    python run_subtask3.py                    # full run (long: see --max-seconds)
    python run_subtask3.py --quick --toy      # 1-2 minute smoke test on a synthetic corpus
Outputs: results/ (report.md, metrics.json, models, SGNS embeddings, figures/)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from s2s import plots  # noqa: E402
from s2s.baselines import extractive_baseline, retrieval_baseline  # noqa: E402
from s2s.data_dialogue import (build_dialogue_data, conv_stats, load_conversations, make_toy_dog,  # noqa: E402
                               mix_bucket)
from s2s.embeddings import nearest, train_sgns  # noqa: E402
from s2s.pipeline_dialogue import (generate, make_model, score, score_by_bucket, src_batch,  # noqa: E402
                                   train_dialogue_model)
from s2s.pipeline_translation import md_table  # noqa: E402
from s2s.tokenize import detokenize  # noqa: E402
from s2s.train import eval_loss, save_json  # noqa: E402
from s2s.train import collate  # noqa: E402
from s2s.decoding import generate_batch  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder with local dialogue files (default: download festvox/cmu_hinglish_dog)")
    ap.add_argument("--docs", default=None, help="JSON file / folder with the Wikipedia documents, if the rows carry none")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--toy", action="store_true", help="SYNTHETIC corpus, smoke test only")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--attn", default="bahdanau", choices=["bahdanau", "luong_general", "luong_dot", "none"])
    ap.add_argument("--hist-turns", type=int, default=3)
    ap.add_argument("--max-hist-len", type=int, default=60)
    ap.add_argument("--max-resp-len", type=int, default=30)
    ap.add_argument("--max-doc-len", type=int, default=100)
    ap.add_argument("--max-vocab", type=int, default=8000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-eval", type=int, default=800)
    ap.add_argument("--emb-dim", type=int, default=100)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=1800, help="time budget per trained model")
    ap.add_argument("--sgns-epochs", type=int, default=5)
    ap.add_argument("--no-sgns", action="store_true", help="random embedding init instead of from-scratch skip-gram")
    ap.add_argument("--ablate-emb", action="store_true", help="also train a grounded model with random embeddings")
    ap.add_argument("--beam", type=int, default=3, help="also decode with beam search of this width (0 = off)")
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--min-len", type=int, default=4, help="minimum reply length when decoding (stops 2-word generic replies)")
    ap.add_argument("--add-english", action="store_true", help="also train on the English version of every training turn (doubles the data)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.max_train, args.max_eval, args.hid, args.emb_dim, args.sgns_epochs = 3, 1500, 100, 64, 48, 2
    figd = os.path.join(args.out, "figures")
    os.makedirs(figd, exist_ok=True)
    t_all = time.time()

    # ------------------------------------------------------------------ 1. data
    if args.toy:
        convs, source = make_toy_dog(500, args.seed), "SYNTHETIC toy corpus (smoke test only)"
    else:
        convs, source = load_conversations(args.data, args.docs, args.data_dir, args.seed)
    cstats = {k: conv_stats([c for c in convs if c["split"] == k]) for k in ("train", "val", "test")}
    data = build_dialogue_data(convs, args.max_train, args.max_eval, args.seed, args.min_freq, args.max_vocab,
                               hist_turns=args.hist_turns, max_hist_len=args.max_hist_len,
                               max_resp_len=args.max_resp_len, max_doc_len=args.max_doc_len, add_english=args.add_english)
    v, ex = data["vocab"], data["examples"]
    st = data["stats"]
    print(f"[data] {source}\n[data] examples train/val/test = {st['train']}/{st['val']}/{st['test']}  vocab {st['vocab']}  "
          f"test reply OOV {st['resp_oov_test']:.1%}")
    v.save(os.path.join(args.out, "vocab.json"))

    # ------------------------------------------------------------------ 2. embeddings from scratch (skip-gram, negative sampling)
    emb, nn_rows = None, []
    if not args.no_sgns:
        sents = [v.encode(t) for t in data["train_texts"]]
        emb = train_sgns(sents, len(v), dim=args.emb_dim, window=4, negatives=5, epochs=args.sgns_epochs, seed=args.seed)
        np.save(os.path.join(args.out, "sgns_embeddings.npy"), emb)
        probes = [w for w in ("movie", "film", "director", "acting", "story", "accha", "acha", "bahut", "kaun", "kya", "dekhi", "hai")
                  if w in v.stoi][:8]
        for w in probes:
            nn_rows.append({"word": w, "nearest neighbours": ", ".join(f"{n} ({s:.2f})" for n, s in nearest(emb, v, w, 5))})

    # ------------------------------------------------------------------ 3. baselines
    test = ex["test"]
    bl_hyps = {"Retrieval (TF-IDF reply picker)": retrieval_baseline(ex["train"], test),
               "Extractive (best document sentence)": extractive_baseline(test, ex["train"])}
    results, hyps_all = {}, {}
    for name, h in bl_hyps.items():
        results[name] = score(test, h); hyps_all[name] = h

    # ------------------------------------------------------------------ 4. neural models
    hists, models = {}, {}
    def build(name, n_src, use_emb):
        m = make_model(args, len(v), n_src, seed=args.seed, emb_matrix=emb if use_emb else None)
        hists[name] = train_dialogue_model(name, m, data, args, args.out)
        models[name] = m
    build("ungrounded", 1, emb is not None)
    build("grounded", 2, emb is not None)
    if args.ablate_emb and emb is not None:
        build("grounded_random_emb", 2, False)
    plots.plot_histories({k: h for k, h in hists.items()}, os.path.join(figd, "loss_curves.png"))

    names = {"ungrounded": "Seq2seq, history only (ungrounded)", "grounded": "Seq2seq, history + document (grounded)",
             "grounded_random_emb": "Grounded, random embeddings"}
    nll = {}
    for key, m in models.items():
        n_src = m.cfg.n_sources
        col = lambda b, n=n_src: collate(b, v.pad, v.bos, v.eos, n)
        nll[key] = eval_loss(m, test, col)
        hy = generate(m, v, test, "greedy", max_len=args.max_resp_len + 5, no_repeat_ngram=args.no_repeat_ngram, min_len=args.min_len)
        results[names[key]] = {**score(test, hy), "test_nll": nll[key][0], "test_ppl": nll[key][1]}
        hyps_all[names[key]] = hy
        print(f"[result] {names[key]:42s} BLEU-1 {results[names[key]]['bleu1']:.2f}  BLEU-2 {results[names[key]]['bleu2']:.2f}  "
              f"ROUGE-L {results[names[key]]['rougeL']:.2f}  ppl {nll[key][1]:.2f}")
    beam_rows = {}
    if args.beam and args.beam > 1:
        for key in ("ungrounded", "grounded"):
            hy = generate(models[key], v, test, "beam", beam=args.beam, max_len=args.max_resp_len + 5, no_repeat_ngram=args.no_repeat_ngram, min_len=args.min_len)
            beam_rows[names[key] + f" + beam {args.beam}"] = score(test, hy)
            hyps_all[names[key] + f" + beam {args.beam}"] = hy
    sample_rows = {}
    for key in ("grounded",):
        hy = generate(models[key], v, test, "sample", temperature=0.8, top_p=0.9, max_len=args.max_resp_len + 5,
                      no_repeat_ngram=args.no_repeat_ngram, min_len=args.min_len, seed=args.seed)
        sample_rows[names[key] + " + top-p 0.9, T=0.8"] = score(test, hy)
        hyps_all[names[key] + " + top-p 0.9, T=0.8"] = hy

    # ------------------------------------------------------------------ 5. does the model USE the document? (mismatched-document test)
    rng = np.random.default_rng(args.seed)
    others = [e["srcs"][1] for e in test]
    perm = rng.permutation(len(test))
    shuffled = [others[perm[i]] if test[perm[i]]["conv"] != test[i]["conv"] else others[(perm[i] + 7) % len(test)] for i in range(len(test))]
    hy_sh = generate(models["grounded"], v, test, "greedy", max_len=args.max_resp_len + 5, no_repeat_ngram=args.no_repeat_ngram, min_len=args.min_len,
                     doc_override=shuffled)
    sh_score = score(test, hy_sh)
    g = results[names["grounded"]]

    # ------------------------------------------------------------------ 6. code-mixing analysis
    bucket_tabs = {}
    for nm in (names["ungrounded"], names["grounded"], "Retrieval (TF-IDF reply picker)"):
        bucket_tabs[nm] = score_by_bucket(test, hyps_all[nm], mix_bucket)
    # ------------------------------------------------------------------ 7. qualitative
    ex_rows = []
    for i in rng.permutation(len(test))[:6]:
        e = test[i]
        ex_rows.append({"field": "history", "text": detokenize(e["hist_tok"])[-220:]})
        ex_rows.append({"field": "document (start)", "text": detokenize(e["doc_tok"])[:200]})
        ex_rows.append({"field": "reference", "text": detokenize(e["resp_tok"])})
        for nm in (names["ungrounded"], names["grounded"], "Retrieval (TF-IDF reply picker)"):
            ex_rows.append({"field": nm.split(",")[0].split("(")[0].strip() if nm != names["grounded"] else "grounded", "text": detokenize(hyps_all[nm][i])})
        ex_rows.append({"field": "", "text": ""})
    # attention over the document for one example
    heat = None
    if models["grounded"].cfg.attn != "none":
        i = next((k for k in range(len(test)) if 4 <= len(test[k]["resp_tok"]) <= 12), 0)
        outs, attn = generate_batch(models["grounded"], src_batch([test[i]], v, 2), v.bos, v.eos, max_len=args.max_resp_len,
                                    no_repeat_ngram=args.no_repeat_ngram, min_len=args.min_len, return_attention=True)
        if outs[0]:
            A = np.array([a[1][:len(test[i]["doc_tok"])] for a in attn[0]])
            L = min(A.shape[1], 40)
            plots.plot_attention(A[:, :L], test[i]["doc_tok"][:L], v.decode(outs[0]), os.path.join(figd, "attention_document.png"),
                                 "Decoder attention over the grounding document")
            heat = "attention_document.png"
    show = [k for k in (names["ungrounded"], names["grounded"], "Retrieval (TF-IDF reply picker)", "Extractive (best document sentence)") if k in results]
    plots.plot_bars([k.split("(")[0].strip()[:28] for k in show], [results[k]["rougeL"] for k in show], os.path.join(figd, "rouge_l.png"),
                    "ROUGE-L vs reference replies (test)", "ROUGE-L")
    keys = ["bleu1", "bleu2", "bleu", "rouge1", "rouge2", "rougeL", "distinct1", "distinct2", "avg_len", "grounding_overlap", "doc_word_hit", "top5_reply_share"]
    rows_main = [{"system": k, **{c: results[k].get(c, float("nan")) for c in keys}, "test ppl": results[k].get("test_ppl", float("nan"))} for k in results]
    extra = {**beam_rows, **sample_rows}
    rows_dec = [{"system": k, **{c: s[c] for c in keys}} for k, s in extra.items()]
    ref_row = {"system": "(reference replies)", "avg_len": float(np.mean([len(e["resp_tok"]) for e in test])),
               "distinct1": __import__("s2s.metrics", fromlist=["x"]).distinct_n([e["resp_tok"] for e in test], 1),
               "distinct2": __import__("s2s.metrics", fromlist=["x"]).distinct_n([e["resp_tok"] for e in test], 2)}
    metrics_json = {"results": results, "decoding_extra": extra, "shuffled_doc": sh_score, "stats": st, "conv_stats": cstats,
                    "config": vars(args), "source": source, "hists": {k: {"n_params": h["n_params"], "best_val_loss": h["best_val_loss"],
                    "train_seconds": h["train_seconds"]} for k, h in hists.items()}}
    save_json(metrics_json, os.path.join(args.out, "metrics.json"))

    # ------------------------------------------------------------------ 8. report
    def pct(a, b):
        return f"{(a - b):+.2f}"
    gu, ug = results[names["grounded"]], results[names["ungrounded"]]
    verdict = (f"The grounded model improves on the ungrounded one by {pct(gu['bleu2'], ug['bleu2'])} BLEU-2, {pct(gu['rougeL'], ug['rougeL'])} ROUGE-L and "
               f"{pct(gu['grounding_overlap'], ug['grounding_overlap'])} grounding overlap."
               if gu["rougeL"] > ug["rougeL"] else
               f"On this run the grounded model does NOT beat the ungrounded one on ROUGE-L ({gu['rougeL']:.2f} vs {ug['rougeL']:.2f}); "
               "the document encoder adds parameters but, with this little data, the model may not have learned to use it.")
    collapsed = [n for n in (names["ungrounded"], names["grounded"]) if results[n].get("top5_reply_share", 0) >= 0.8]
    collapse_note = (("\n\n**Warning - degenerate generation.** " + " and ".join(collapsed) + " output one of 5 generic replies for >= 80% of the "
                      "test turns, so the comparison above (and the mismatched-document test below) says little about document use: the models "
                      "collapsed to the most frequent short replies because the training set is tiny (a few thousand real reply turns). "
                      + ("`--add-english` was already used in this run, so the remaining options are a smaller `--max-vocab`, more dropout, "
                         "subword units or a copy mechanism; report this as a finding. " if args.add_english else
                         "Re-run with `--add-english`, a smaller `--max-vocab` and/or more regularisation. ")
                      + "Treat the sampled-decoding rows and the retrieval baseline as the more informative comparison.")
                     if collapsed else "")
    drop = (f"With mismatched documents the grounded model's ROUGE-L goes from {g['rougeL']:.2f} to {sh_score['rougeL']:.2f} and the "
            f"share of reply words found in the (wrong) document from {g['grounding_overlap']:.2f} to {sh_score['grounding_overlap']:.2f}: "
            + ("it reads the document." if g["rougeL"] - sh_score["rougeL"] > 0.5 or g["grounding_overlap"] - sh_score["grounding_overlap"] > 0.03
               else "little change, so the document is used only weakly (or, if generation has collapsed, not measurably at all)."))
    bkt = bucket_tabs[names["grounded"]]
    bk_text = ""
    if len(bkt) >= 2:
        by = {r["bucket"]: r for r in bkt}
        eng, hin = by.get("mostly English"), by.get("mostly Hindi")
        if eng and hin:
            bk_text = (f"Mostly-English turns score ROUGE-L {eng['ROUGE-L']:.1f} (n={eng['n']}) vs {hin['ROUGE-L']:.1f} for mostly-Hindi turns "
                       f"(n={hin['n']}); " + ("heavily code-mixed turns are harder, as expected: romanised Hindi has inconsistent spelling, so more word types and more <unk>."
                                               if hin["ROUGE-L"] < eng["ROUGE-L"] else "no degradation on Hindi-heavy turns in this run.")
                       + (" (Caution: the neural models have collapsed to generic replies, so this bucket comparison mostly reflects reply length, not code-mixing ability.)" if collapsed else ""))
    warn = "> **WARNING: synthetic toy corpus - smoke test only. Do not report these numbers.**\n\n" if args.toy else ""
    cs = cstats["train"]
    report = f"""# Sub-task 3 report: document-grounded Hinglish dialogue generation

{warn}## Data and preprocessing
{source}. Conversations: train {cstats['train']['conversations']:,} / val {cstats['val']['conversations']:,} / test {cstats['test']['conversations']:,}
(turns {cs['turns']:,} in train). Each training example = one reply turn, its previous <= {args.hist_turns} turns (speaker-tagged `<usr1>`/`<usr2>`,
cut to {args.max_hist_len} tokens) and the grounding document section the speaker was reading (<= {args.max_doc_len} tokens).
Examples used: train {st['train']:,}, val {st['val']:,}, test {st['test']:,} (capped for CPU time){' - training also uses the English version of every training turn (--add-english); val/test are Hinglish only' if args.add_english else ''}.
Decoding uses a minimum reply length of {args.min_len} tokens.

Custom preprocessing for code-mixed text: Unicode NFC, lower-casing, URL tag, collapse of letter elongation (`yaaaar` -> `yaar`),
Devanagari-aware tokenizer (a naive `\\w+` splits Devanagari words at vowel signs), emoji/punctuation kept as tokens, one shared
vocabulary of {st['vocab']:,} types built on the training conversations only (min frequency {args.min_freq}); test replies have {st['resp_oov_test']:.1%} out-of-vocabulary tokens.
Share of Devanagari tokens in the Hinglish text: {cs['devanagari_token_share']:.1%} ({'so the Hinglish text is essentially romanised (Latin script)' if cs['devanagari_token_share'] < 0.05 else 'so a noticeable part of the Hinglish text is written in Devanagari and the rest in Latin script'}).
Code-mixing proxy: share of a turn's Latin tokens that also occur in its English translation ({'mean ' + format(cs['english_share_mean'], '.2f') if cs['english_share_mean'] is not None else 'unavailable'}); turns are bucketed
mostly English (>=0.6) / mixed / mostly Hindi (<0.3; Devanagari words count as Hindi). Train buckets: {cs['buckets']}.

## Embeddings from scratch
{'Skip-gram with negative sampling trained only on the training dialogue turns and the grounding documents (dimension ' + str(args.emb_dim) + ', ' + str(args.sgns_epochs) + ' epochs), used to initialise the shared embedding table (fine-tuned afterwards). No pretrained vectors.' if emb is not None else 'Random initialisation (--no-sgns).'}

{md_table(nn_rows, ['word', 'nearest neighbours']) if nn_rows else ''}

## Model
Two LSTM encoders (history, document) share one embedding table with the decoder; their final states are combined by a linear bridge into the
decoder's initial state. At every step the decoder attends over **both** encoders' outputs separately ({args.attn} attention), concatenates the two
contexts with its own state and predicts the next token. Ungrounded baseline = the identical network without the document encoder.
{make_desc(models)}
Other baselines: TF-IDF retrieval of the training reply whose history is most similar (ungrounded), and extractive selection of the best-matching document sentence (grounded, non-generative).

## Results (test, greedy decoding with no-repeat-{args.no_repeat_ngram}-gram blocking)
{md_table(rows_main, ['system', 'bleu1', 'bleu2', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'distinct2', 'avg_len', 'grounding_overlap', 'doc_word_hit', 'top5_reply_share', 'test ppl'])}

`grounding_overlap` = share of a reply's content words that appear in the document; `doc_word_hit` = share of replies with >= 1 document word;
`top5_reply_share` = share of outputs that are one of the 5 most common replies (high = generic replies). Reference replies: average length {ref_row['avg_len']:.1f}, distinct-1 {ref_row['distinct1']:.3f}, distinct-2 {ref_row['distinct2']:.3f}.

{verdict}{collapse_note}

![rouge](figures/rouge_l.png)
![loss](figures/loss_curves.png)

### Decoding variants for the neural models
{md_table(rows_dec, ['system', 'bleu1', 'bleu2', 'rougeL', 'distinct1', 'distinct2', 'avg_len', 'grounding_overlap']) if rows_dec else ''}

### Does the model actually use the document?
{drop}

{f'![attention](figures/{heat})' if heat else ''}

## Code-mixed vs mostly-English turns
{chr(10).join(f'**{nm}**' + chr(10) + chr(10) + md_table(t, ['bucket', 'n', 'BLEU-1', 'BLEU-2', 'ROUGE-L', 'avg ref len', 'avg gen len']) + chr(10) for nm, t in bucket_tabs.items())}
{bk_text}

## Qualitative examples
{md_table(ex_rows, ['field', 'text'])}

## Design decisions and limitations
- Word-level vocabulary with `<unk>` (romanised Hindi spelling varies a lot; subword units would help and are the natural next step).
- Attention is computed separately for history and document so the model can balance them; a copy mechanism would let it reproduce rare document words (names, years) that are `<unk>` or rare in the vocabulary.
- Training is capped ({st['train']:,} examples used, {args.max_seconds:.0f} s per model) so it runs on a CPU; absolute scores are therefore low and BLEU against a single reference reply is a weak signal for open-ended dialogue - hence ROUGE, diversity, grounding overlap and the mismatched-document test.
- No pretrained models or packaged seq2seq pipelines were used; all components are implemented from scratch in NumPy.

Run time: {time.time() - t_all:.0f} s.
"""
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[done] report -> {os.path.join(args.out, 'report.md')}")


def make_desc(models):
    return "Parameters: " + ", ".join(f"{k} {m.n_params():,}" for k, m in models.items()) + "."


if __name__ == "__main__":
    main()
