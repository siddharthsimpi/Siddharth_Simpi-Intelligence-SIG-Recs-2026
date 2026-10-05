# Part C — Image-Based Product Matching

**Files:** `notebook.py` → `notebook.ipynb`, `results/` (`experiments.csv/.md`, `nn_metrics.csv`, `error_rates_by_condition.csv`, figures incl. nearest-neighbour grids, `summary.json`).

## Experiments
| ID | Model | dim | Similarity |
|---|---|---|---|
| Baseline 0 | pHash (64-bit, from the CSV) | 64 | Hamming ≈ cosine |
| Baseline | ResNet-50 (ImageNet) GAP | 2048 | cosine |
| Exp 1 | EfficientNet-B0 | 1280 | cosine |
| Exp 2 | CLIP ViT-B/32 image encoder | 512 | cosine |
| Exp 3 | ResNet-50 + PCA-whitening (fit on train only) | 256 | cosine |
| Exp 4 | ResNet-50 raw | 2048 | dot product / Euclidean |

Reported per experiment: model, embedding dim, similarity, threshold, **computational cost** (embedding time, memory MB, exact-search time), matching performance (pair AUC/AP/F1 + Kaggle-style retrieval-F1).

Also: preprocessing (resize whole image, no crop, per-backbone normalisation, robust to missing files), embedding concept + similarity distributions, threshold analysis, **nearest-neighbour analysis** (query + top-5 grids, hit@1 / precision@5 / recall@10), **error analysis** with measured error rates per condition (different aspect ratio, background, low contrast, low resolution, pHash far/near) and a failure taxonomy, computational discussion (chunked exact search, optional faiss HNSW vs exact).

## Run
```bash
cd PartC && python notebook.py       # GPU recommended for the first embedding pass; embeddings are cached in ../cache
```
## External resources
torchvision ResNet-50 / EfficientNet-B0 weights, OpenAI CLIP ViT-B/32 (HuggingFace `transformers`), optional `faiss-cpu`.
