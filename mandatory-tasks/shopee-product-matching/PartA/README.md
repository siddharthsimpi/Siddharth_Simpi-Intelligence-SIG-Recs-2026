# Part A — Dataset Exploration

**Files:** `notebook.py` (source, percent format) → `notebook.ipynb` (built with `tools/build_notebooks.py`), `results/` (figures + `dataset_summary.json`, `group_size_distribution.csv`).

## What the notebook does
1. **Understand the dataset** – columns/meaning, what a product group is (`label_group` = equivalence class), how matching is represented, how we split (by group → no leakage).
2. **Statistics** – #listings, #groups, group-size distribution, unique images/pHashes, duplicate images (and images shared by *different* products), duplicate/near-duplicate titles, title-length, language/script heuristic, ALL-CAPS/emoji/short-title noise, image size/aspect/contrast.
3. **Similarity case studies (with images)** – same product / different images; similar titles / different products; near-identical images / different products; noisy & incomplete titles; same product with disjoint titles; number tokens as variant markers. Plus quantified separability of the two cheapest signals (title char-TF-IDF cosine, pHash Hamming distance).
4. **Challenges table** (10 challenges, each linked to evidence) and **answers to the guiding questions**.

## Run
```bash
cd PartA && python notebook.py          # or open notebook.ipynb
```
## Notes / honesty
* The "Observations" markdown cells are written as interpretation templates. **Re-read them against your own printed numbers and edit** – you must be able to defend every statement.
* No external code was used except pandas/matplotlib/scikit-learn.
