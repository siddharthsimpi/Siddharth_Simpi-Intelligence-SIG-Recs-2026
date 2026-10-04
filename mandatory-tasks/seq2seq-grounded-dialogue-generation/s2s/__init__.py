"""s2s: sequence-to-sequence toolkit built from scratch (NumPy only).

autograd  - tiny reverse-mode autodiff + fused LSTM / attention / cross-entropy ops (gradient-checked)
models    - encoder-decoder (optionally with attention, optionally with two conditioning sources)
decoding  - greedy, beam search, temperature / top-k / top-p sampling
tokenize  - Devanagari-aware, code-mixing-aware tokenisation and vocabularies
embeddings- skip-gram with negative sampling trained from scratch
metrics   - BLEU, ROUGE, distinct-n, grounding overlap
No deep-learning framework, no pretrained weights, no packaged seq2seq pipeline is used anywhere.
"""
