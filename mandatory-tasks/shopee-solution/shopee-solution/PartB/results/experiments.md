| Experiment | Representation | Similarity | dim | pair_thr | AUC | AP | pair_F1 | retr_thr | retr_F1 | cand_R | embed_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF word 1-gram | cosine | 11673 | 0.2014 | 0.9625 | 0.9644 | 0.9041 | 0.3477 | 0.8507 | 0.9802 | 0.1400 |
| Exp 1a | TF-IDF word 1-2gram | cosine | 55186 | 0.0887 | 0.9614 | 0.9630 | 0.9045 | 0.1807 | 0.8391 | 0.9802 | 0.1800 |
| Exp 2 | TF-IDF char_wb 2-5 | cosine | 44916 | 0.2609 | 0.9672 | 0.9707 | 0.9109 | 0.3743 | 0.8479 | 0.9849 | 0.6300 |
| Exp 3 | SBERT multilingual MiniLM | cosine | 384 | 0.4246 | 0.9430 | 0.9436 | 0.8757 | 0.7162 | 0.6961 | 0.8981 | 111.1800 |
| Exp 4a | SBERT (raw, un-normalised) | dot product | 384 | 5.3246 | 0.9159 | 0.9128 | 0.8484 | 12.0147 | 0.5959 | 0.8716 |  |
| Exp 4b | SBERT (raw, un-normalised) | euclidean (neg.) | 384 | -4.1045 | 0.9059 | 0.9086 | 0.8299 | -2.6429 | 0.6096 | 0.8510 |  |
| Exp 5 | char TF-IDF ⊕ SBERT | cosine (mean of both) | 45300 | 0.3385 | 0.9710 | 0.9724 | 0.9084 | 0.4989 | 0.8409 | 0.9758 |  |
| Exp 6 | word sets | Jaccard |  | 0.1154 | 0.9610 | 0.9625 | 0.9025 |  |  |  |  |