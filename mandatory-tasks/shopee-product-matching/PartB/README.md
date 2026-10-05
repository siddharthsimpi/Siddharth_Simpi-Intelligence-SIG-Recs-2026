# Part B — Text-Based Product Matching

**Files:** `notebook.py` → `notebook.ipynb`, `results/` (`experiments.csv/.md`, `threshold_table.csv`, `error_examples.csv`, figures, `summary.json`).

## Protocol
Group-wise train/val/test split. **Thresholds are chosen on val, every reported number is on test.** Two views of performance:
* *Pair benchmark* (fixed for Parts B/C/Finale): positives + random / text-hard / image-hard negatives → AUC, AP, P/R/F1.
* *Retrieval metric* (Kaggle): per-listing F1 of the predicted match set from top-50 neighbours.

## Experiments (all documented with method / why / implementation / metric / results / observations in the notebook)
| ID | Representation | Similarity |
|---|---|---|
| Baseline | TF-IDF word 1-gram | cosine |
| Exp 1 | TF-IDF word 1–2-gram | cosine |
| Exp 2 | TF-IDF char_wb 2–5-gram | cosine |
| Exp 3 | multilingual SBERT (MiniLM) | cosine |
| Exp 4 | SBERT raw vectors | dot / Euclidean (does normalisation matter?) |
| Exp 5 | char TF-IDF ⊕ SBERT hybrid | cosine (mean) |
| Exp 6 | word sets | Jaccard |

Also: **threshold analysis** (P/R/F1, FP/FN counts vs threshold, PR curve, retrieval-F1 vs threshold, how the final threshold is chosen) and **error analysis** (TP / FP / FN examples, error rates by type, which method uniquely finds which matches).

## Run
```bash
cd PartB && python notebook.py
```
## External resources
scikit-learn `TfidfVectorizer`; `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Reimers & Gurevych, 2019).
Edit the *Observations / Conclusions* cells with your own findings after running.
