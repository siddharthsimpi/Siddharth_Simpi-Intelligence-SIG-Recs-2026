# Sub-task 3 solution notes (assignment text: README.md)

Run: `python inspect_dataset.py` then `python run_subtask3.py` (add `--quick --toy` for a smoke test). Results: `results/report.md`.

1. **Data loading / cleaning:** schema-adaptive loader for `festvox/cmu_hinglish_dog` (per-turn rows or conversation-level records, `translation.en` / `translation.hi_en`,
   speaker ids, `docIdx`, `wikiDocumentIdx`); grounding documents from the rows, from `--docs`, or from the original CMU DoG repository.
2. **Custom preprocessing:** NFC, lower-casing, URL tag, elongation collapsing, Devanagari-aware tokenizer, speaker tags, one shared vocabulary built from training
   conversations only; code-mixing proxy (share of Latin tokens that also appear in the English translation) used to bucket turns.
3. **Embeddings from scratch:** skip-gram with negative sampling (`s2s/embeddings.py`) trained on training turns + documents, used to initialise the shared embedding table.
4. **Generator:** two LSTM encoders (history, document) -> linear bridge -> LSTM decoder that attends over both encoders at every step (`Seq2Seq(n_sources=2)`).
5. **Baselines:** history-only seq2seq (same network, no document encoder), TF-IDF retrieval of a training reply, extractive document-sentence picker.
6. **Evaluation:** BLEU-1/2/4, ROUGE-1/2/L, distinct-n, perplexity, grounding overlap, generic-reply share, a mismatched-document control (does the model really read the document?),
   per-bucket scores for mostly-English vs heavily code-mixed turns, qualitative examples and a document-attention heat-map.
