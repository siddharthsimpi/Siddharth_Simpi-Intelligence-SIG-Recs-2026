# From translation to grounded dialogue generation: short report

## 1. Approach and datasets
All models are built from scratch in NumPy: a small reverse-mode autograd with fused LSTM, attention and cross-entropy ops (verified against finite differences), an LSTM encoder-decoder, Bahdanau / Luong attention, greedy / beam / sampling decoders, BLEU / ROUGE and skip-gram embeddings. No pretrained models and no packaged seq2seq pipeline are used.

- **Sub-tasks 1-2 (translation, en -> hi):** train/val/test = 20000/541/1000 pairs, vocabularies 6000/6000 (word level, script-aware tokenizer).
- **Sub-task 3 (grounded Hinglish dialogue):** HuggingFace festvox/cmu_hinglish_dog; 14,748/800/800 train/val/test reply examples, one shared vocabulary of 4,000 types, embeddings trained from scratch with skip-gram negative sampling.

## 2. Experimental results

**Sub-task 1 - basic encoder-decoder (no attention):** test BLEU **2.83** (greedy). By source length: 1-5: 10.2, 6-10: 4.2, 11-15: 2.7, 16-20: 2.5, 21+: 2.0.

**Sub-task 2 - attention (test BLEU, greedy):**

| model | best val nll | test BLEU |
|---|---|---|
| no attention | 3.68 | 3.45 |
| Bahdanau (additive) | 3.39 | 6.93 |
| Luong (general) | 3.70 | 4.36 |

**Decoding strategies on Bahdanau (additive):**

| strategy | BLEU | distinct-2 | avg len |
|---|---|---|---|
| greedy | 6.93 | 0.17 | 13.39 |
| beam 3 | 7.80 | 0.17 | 12.68 |
| beam 5 | 7.82 | 0.17 | 12.41 |
| sampling T=0.7 | 4.33 | 0.38 | 13.55 |
| sampling T=1.0 | 2.29 | 0.60 | 13.65 |
| top-k 10 (T=1) | 3.51 | 0.32 | 13.88 |
| top-p 0.9 (T=1) | 3.07 | 0.52 | 13.68 |

**Sub-task 3 - dialogue generation (test):**

| system | BLEU-2 | ROUGE-L | distinct-2 | grounding overlap |
|---|---|---|---|---|
| Retrieval (TF-IDF reply picker) | 1.73 | 5.74 | 0.62 | 0.02 |
| Extractive (best document sentence) | 0.77 | 3.77 | 0.20 | 1.00 |
| Seq2seq, history only (ungrounded) | 1.29 | 5.83 | 0.00 | 0.00 |
| Seq2seq, history + document (grounded) | 1.63 | 9.88 | 0.01 | 0.00 |

Mismatched-document test: ROUGE-L 9.88 -> 9.74, grounding overlap 0.00 -> 0.00.

## 3. Insights and takeaways

- Attention changes test BLEU from 3.45 to 6.93 (Bahdanau (additive)): the decoder no longer depends on one fixed vector and can look back at the relevant source words (see the attention heat-maps in `subtask2/results/figures`).
- Decoding: beam search +0.88 BLEU vs greedy; sampling lowers BLEU (2.3) but raises diversity (distinct-2 0.60 vs 0.17).
- Grounding must be tested, not assumed: BLEU alone is weak for open-ended replies, so the report adds ROUGE, diversity, document-overlap and a mismatched-document control; retrieval and extractive baselines show what a non-generative system achieves.
- Code-mixed (romanised Hindi) text has inconsistent spelling, so vocabulary and `<unk>` rates are higher; the per-bucket tables in `subtask3/results/report.md` quantify the degradation.
- Limits: small CPU-trained models, word-level vocabulary and no copy mechanism; subword units and copy attention are the next steps.

Detailed tables, figures and example outputs: `subtask1/results/report.md`, `subtask2/results/report.md`, `subtask3/results/report.md`.
