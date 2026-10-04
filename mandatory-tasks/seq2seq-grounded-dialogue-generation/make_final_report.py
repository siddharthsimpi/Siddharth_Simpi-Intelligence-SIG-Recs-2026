#!/usr/bin/env python
"""Assemble FINAL_REPORT.md (the short 2-3 page write-up) from the three sub-tasks' results/metrics.json files.

    python make_final_report.py
Run it after run_subtask1.py, run_subtask2.py and run_subtask3.py (missing parts are marked as not run).
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(sub):
    p = os.path.join(ROOT, sub, "results", "metrics.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def table(rows, cols, fmt="{:.2f}"):
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt.format(r[c]) if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    m1, m2, m3 = load("subtask1"), load("subtask2"), load("subtask3")
    toy = any(m and "SYNTHETIC" in str(m.get("config", {}).get("toy", "")) or (m and m.get("config", {}).get("toy")) for m in (m1, m2, m3))
    L = ["# From translation to grounded dialogue generation: short report", ""]
    if toy:
        L += ["> **WARNING: at least one sub-task was run with `--toy` (synthetic data). Re-run without it before reporting numbers.**", ""]
    L += ["## 1. Approach and datasets",
          "All models are built from scratch in NumPy: a small reverse-mode autograd with fused LSTM, attention and cross-entropy ops "
          "(verified against finite differences), an LSTM encoder-decoder, Bahdanau / Luong attention, greedy / beam / sampling decoders, "
          "BLEU / ROUGE and skip-gram embeddings. No pretrained models and no packaged seq2seq pipeline are used.", ""]
    if m1:
        s = m1["stats"]
        cfg = m1.get("config", {})
        pair = f"{cfg.get('src_lang') or 'en'} -> {cfg.get('tgt_lang') or 'hi'}"
        L += [f"- **Sub-tasks 1-2 (translation, {pair}):** train/val/test = {s['n_train']}/{s['n_val']}/{s['n_test']} pairs, vocabularies "
              f"{s['src_vocab']}/{s['tgt_vocab']} (word level, script-aware tokenizer)."]
    if m3:
        s = m3["stats"]
        L += [f"- **Sub-task 3 (grounded Hinglish dialogue):** {m3['source']}; {s['train']:,}/{s['val']:,}/{s['test']:,} train/val/test reply examples, "
              f"one shared vocabulary of {s['vocab']:,} types, embeddings trained from scratch with skip-gram negative sampling."]
    L += ["", "## 2. Experimental results", ""]
    if m1:
        L += [f"**Sub-task 1 - basic encoder-decoder (no attention):** test BLEU **{m1['test']['bleu']:.2f}** (greedy). By source length: " +
              ", ".join(f"{r['bucket']}: {r['bleu']:.1f}" for r in m1["by_length"]) + ".", ""]
    if m2:
        L += ["**Sub-task 2 - attention (test BLEU, greedy):**", "", table(m2["models"], ["model", "best val nll", "test BLEU"]), "",
              f"**Decoding strategies on {m2['best_attention_model']}:**", "",
              table(m2["decoding"], ["strategy", "BLEU", "distinct-2", "avg len"]), ""]
    if m3:
        rows = []
        for k, r in m3["results"].items():
            rows.append({"system": k, "BLEU-2": r["bleu2"], "ROUGE-L": r["rougeL"], "distinct-2": r["distinct2"],
                         "grounding overlap": r.get("grounding_overlap", float("nan"))})
        L += ["**Sub-task 3 - dialogue generation (test):**", "", table(rows, ["system", "BLEU-2", "ROUGE-L", "distinct-2", "grounding overlap"]), ""]
        sh = m3["shuffled_doc"]
        g = next((r for k, r in m3["results"].items() if "grounded" in k and "ungrounded" not in k and "random" not in k), None)
        if g:
            L += [f"Mismatched-document test: ROUGE-L {g['rougeL']:.2f} -> {sh['rougeL']:.2f}, grounding overlap {g['grounding_overlap']:.2f} -> {sh['grounding_overlap']:.2f}.", ""]
    L += ["## 3. Insights and takeaways", ""]
    if m1 and m2:
        base = next(r for r in m2["models"] if r["model"] == "no attention")["test BLEU"]
        best = max((r for r in m2["models"] if r["model"] != "no attention"), key=lambda r: r["test BLEU"])
        L += [f"- Attention changes test BLEU from {base:.2f} to {best['test BLEU']:.2f} ({best['model']}): the decoder no longer depends on one fixed vector "
              "and can look back at the relevant source words (see the attention heat-maps in `subtask2/results/figures`)."]
        d = {r["strategy"]: r for r in m2["decoding"]}
        L += [f"- Decoding: beam search {d['beam 5']['BLEU'] - d['greedy']['BLEU']:+.2f} BLEU vs greedy; sampling lowers BLEU "
              f"({d['sampling T=1.0']['BLEU']:.1f}) but raises diversity (distinct-2 {d['sampling T=1.0']['distinct-2']:.2f} vs {d['greedy']['distinct-2']:.2f})."]
    if m3:
        L += ["- Grounding must be tested, not assumed: BLEU alone is weak for open-ended replies, so the report adds ROUGE, diversity, "
              "document-overlap and a mismatched-document control; retrieval and extractive baselines show what a non-generative system achieves.",
              "- Code-mixed (romanised Hindi) text has inconsistent spelling, so vocabulary and `<unk>` rates are higher; the per-bucket tables in "
              "`subtask3/results/report.md` quantify the degradation.",
              "- Limits: small CPU-trained models, word-level vocabulary and no copy mechanism; subword units and copy attention are the next steps."]
    L += ["", "Detailed tables, figures and example outputs: `subtask1/results/report.md`, `subtask2/results/report.md`, `subtask3/results/report.md`.", ""]
    with open(os.path.join(ROOT, "FINAL_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("wrote FINAL_REPORT.md")


if __name__ == "__main__":
    main()
