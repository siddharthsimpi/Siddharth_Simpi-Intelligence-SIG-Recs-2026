# Task 1 solution notes — Collaborative Filtering: memory-based kNN vs. matrix factorization

(The original assignment text is in `README.md`; this file documents the solution.)

Everything is implemented from scratch with **NumPy** (no recommender libraries, no pretrained weights).

## Dataset (own choice, per the task)
**MovieLens 100K** (GroupLens): 943 users x 1,682 movies, 100,000 explicit 1-5 star ratings, every user has >= 20
ratings. Why it fits: it is a classic user-item explicit-feedback benchmark, small enough to run both methods on a
laptop in minutes, dense enough for neighbourhood similarities to be meaningful, and still sparse (~94%) with a
long tail of users/items, which is exactly where memory-based and model-based methods differ.
It is downloaded automatically on first run (with automatic retries for broken Windows/macOS certificate stores). If the download
fails, the script falls back to a clearly labelled **synthetic** MovieLens-like dataset and prints/writes a warning — do not report those numbers as real.
You can also unzip `ml-100k.zip` manually into `data/` so that `data/ml-100k/u.data` exists.

## Layout
```
run_experiment.py      one script: EDA -> tuning -> test evaluation -> plots -> report
cf/data.py             loading, synthetic fallback, per-user split, matrices
cf/memory_cf.py        A) user-user / item-item kNN CF
cf/mf.py               B) biased matrix factorization (Adam or SGD), early stopping
cf/metrics.py          RMSE, MAE, Precision/Recall/NDCG/HitRate/Coverage@K, rated-item NDCG
cf/plots.py            figures
tests/test_cf.py       unit tests (incl. README worked example = 4.47, gradient check)
results/               generated: report.md, metrics.csv, environment.json, figures/ ...
```

## Run it
```
pip install -r ..\requirements.txt
python tests\test_cf.py              (or: python -m pytest -q)   unit tests
python run_experiment.py --quick     ~10 s smoke run
python run_experiment.py             full run, ~1 minute  -> results\report.md
```
Options: `--dataset {auto,movielens,synthetic}`, `--seed 42`, `--out results`, `--data-dir data`.

## Method and decisions
**Split.** Per-user random 80/10/10 train/validation/test (seed 42); each user keeps >= 1 validation and test rating.
Means, similarities and factors use **train only**; hyper-parameters are picked on **validation RMSE**; the test
set is scored once. Seeds, package versions, hardware and the command are saved to `results/environment.json`.

**A) Memory-based CF.** Mean-centred cosine similarity over co-rated entries (Pearson-style; adjusted cosine for
item-item), multiplied by `n_co / (n_co + shrink)` so pairs with only a few co-ratings are discounted; only positive
similarities are kept; prediction = user mean + similarity-weighted average of neighbours' centred ratings, falling
back to the user mean when no neighbour is available. Tuned: k in {10..400}, shrink in {0, 25, 100}.
*Simplification:* neighbours are the global top-k most similar users/items (sparsified similarity matrix), not the
top-k among those who rated the target — it makes prediction two matrix products, at a small accuracy cost.

**B) Matrix factorization.** `r_hat = mu + b_u + b_i + p_u . q_i`, L2-regularised squared loss, analytic gradients
(verified against finite differences in the tests), mini-batch (1024), N(0, 0.1^2) init, early stopping on
validation RMSE (best weights restored). Tuned: factors in {16, 32, 64}, L2 in {0.02, 0.05, 0.1}.
*Why Adam:* per-parameter step sizes cope with the very uneven update frequency of popular vs. rare users/items and
need little learning-rate tuning; the cost is two extra state arrays and occasionally slightly worse generalisation
than carefully tuned SGD. A plain-SGD run is included as an ablation (`03_mf_curves.png`).

**Baselines.** Global mean (random ranking), popularity (ranking only), bias-only model (mu + b_u + b_i).

**Metrics.** RMSE, MAE; `ndcg_rated@5` (does the predicted rating order each user's *rated* test items correctly?);
full-catalogue Precision/Recall/NDCG/HitRate@10 and coverage@10 (relevant = test rating >= 4, seen items excluded).
Full-catalogue top-K is dominated by *exposure* (which items people chose to rate), so the popularity baseline can
beat rating predictors there; report both views.

## When to prefer which (general guidance — confirm with your own `results/report.md`)
* **Memory-based:** no training step, trivially explainable ("because similar users liked it"), new ratings take
  effect immediately, strong with dense data and few items; item-item is stable when items << users. Drawbacks:
  O(n^2) similarity memory, slow/poor on very sparse data, weak for users/items with few ratings.
* **Matrix factorization:** compact model (O((U+I)k)), fast inference (a dot product), generalises through latent
  factors and biases so it usually has lower RMSE on sparse data, and extends naturally to side features and neural
  models (Tasks 2-3). Drawbacks: needs training and tuning, less transparent, needs re-training for new data.

## Notes / limitations
* One random split with one seed; for confidence intervals repeat with several `--seed` values.
* Dense matrices are fine for ML-100K; for much larger data use sparse matrices / approximate neighbours.
