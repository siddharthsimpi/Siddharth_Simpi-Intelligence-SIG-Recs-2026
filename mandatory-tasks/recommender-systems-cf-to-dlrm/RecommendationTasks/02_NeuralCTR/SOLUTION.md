# Task 2 solution notes: neural CTR prediction

(The assignment text is in `README.md`; the generated results with numbers, tables and plots are in `results/report.md`.)

## Run
```
python run_task2.py --quick      # ~1 min smoke run
python run_task2.py              # full run, ~10-15 min on one CPU core
```
Data is found automatically (`..\..\datasets\dataset (tasks 2 and 3).zip` or `train.csv`/`test.csv`), or pass `--data <folder or zip>`.

## What is implemented (all from scratch, NumPy, manual back-propagation; gradient-checked in `tests/`)
- `ctrlib/nn.py`: Linear, ReLU, Dropout, MLP, concatenated embedding tables, Adam, stable BCE-with-logits.
- `ctrlib/models.py`: LogReg, FM, **vanilla MLP** (main model), **DCN** (bonus), DLRM (Task 3).
- Vanilla MLP = concat[13 numeric (+ missing indicators), 26 embeddings] -> ReLU layers (+dropout, + L2) -> logit.

## Decisions and why
| decision | choice | reason / trade-off |
|---|---|---|
| split | stratified 80/20 of `train.csv` into train/validation; `test.csv` untouched | rare positives (3.2%) -> stratify so validation has clicks |
| preprocessing | fit on train part only: median imputation + missing flags, signed log1p, standardise; categories seen < 10 times -> OOV | no leakage; log tames heavy tails; threshold 3/10/30 compared on validation |
| loss | binary cross-entropy, **no re-weighting** | keeps probabilities calibrated (important for CTR); imbalance handled at threshold selection instead |
| optimiser | Adam, lr 3e-4, batch 512 | embeddings of rare categories get sparse noisy gradients; Adam's per-parameter step sizes cope with that. Trade-off: two extra state arrays per parameter and sometimes slightly worse generalisation than tuned SGD |
| regularisation | dropout, weight decay, early stopping (patience 4, restore best) | ~1,000 clicks in 32k rows -> strong over-fitting |
| model selection | lowest **validation log loss**, mean of 2 seeds | proper scoring rule; AUC on ~256 validation clicks is too noisy |
| threshold | max-F1 on validation, frozen for test | accuracy at 0.5 is meaningless (always "no click" = ~96.7%) |
| reporting | 5 seeds, mean +- std, bootstrap CIs | test set has only ~335 clicks |

Metrics: ROC-AUC, PR-AUC, log loss, accuracy (at 0.5 and at the chosen threshold), precision, recall, F1, ECE + reliability diagram.

## Output files (`results/`)
`report.md` (full write-up), `grid.csv` (every configuration), `final_test_metrics.csv`, `selected.json` (re-used by Task 3),
`environment.json` (seeds, versions, hardware, command), `figures/`.
