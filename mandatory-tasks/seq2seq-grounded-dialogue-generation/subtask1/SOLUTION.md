# Sub-task 1 solution notes (assignment text: README.md)

Run: `python run_subtask1.py` (add `--quick --toy` for a smoke test). Results: `results/report.md`.

- **Model:** embedding -> 1-layer LSTM encoder; its final (h, c) is the context vector that initialises a 1-layer LSTM decoder; teacher forcing,
  cross-entropy, Adam, gradient clipping, dropout, LR decay on plateau, early stopping on validation loss. Written from scratch (`s2s/autograd.py`, `s2s/models.py`).
- **Data:** parallel pairs from `data/translation/` (course Drive folder, e.g. `en_fr_train/val/test.csv` = English -> French; Tatoeba fallback). Script-aware tokenizer (Devanagari matras, French accents and elisions), lower-casing, length filter,
  de-duplication, random train/val/test split, vocabularies from train only.
- **Evaluation:** corpus BLEU-4 (own implementation, add-1 smoothing) on a held-out test split; breakdown by source length and by presence of out-of-vocabulary words;
  worst/typical/long-sentence examples. These break-downs are the "where it struggles" write-up.
