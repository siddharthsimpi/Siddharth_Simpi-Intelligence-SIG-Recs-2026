# Sub-task 2 solution notes (assignment text: README.md)

Run: `python run_subtask2.py` (after sub-task 1; add `--quick --toy` for a smoke test). Results: `results/report.md`.

- **Attention:** Bahdanau (additive, previous-state query) and Luong (general bilinear score, input feeding); `--with-dot` adds Luong dot-product.
  Fused forward/backward ops in `s2s/autograd.py`, gradient-checked in `tests/test_autograd.py`.
- **Same data and settings as sub-task 1**; the no-attention model is re-used from sub-task 1 when the settings match.
- **Table:** no attention vs. each attention variant (test BLEU), BLEU by source length, attention heat-maps.
- **Decoding comparison** on the best attention model: greedy, beam 3 / 5 (GNMT length penalty), temperature sampling, top-k, top-p; BLEU, distinct-1/2,
  length and repetition, plus example outputs side by side.
