#!/usr/bin/env python
"""Task 2 -- Neural CTR prediction: vanilla MLP (architecture search) + bonus Deep & Cross Network.

    python run_task2.py            # full run (~5-8 min on 1 CPU core)
    python run_task2.py --quick    # smoke run (~1 min)
Outputs -> results/ : report.md, grid.csv, final_test_metrics.csv, selected.json, environment.json, figures/
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from ctrlib import data as D, plots, specs as S            # noqa: E402
from ctrlib.eda import compute_eda                         # noqa: E402
from ctrlib.experiment import final_eval, md_table, pm, select, summarize   # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # never crash on a legacy Windows console encoding
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="dir with train.csv/test.csv, or the dataset zip")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--seed", type=int, default=42, help="seed of the train/val split")
    ap.add_argument("--select-seeds", type=int, default=2)
    ap.add_argument("--final-seeds", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=10, help="categories seen < this many times in train -> OOV id (chosen on validation: 3/10/30 tried)")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    fig_dir = os.path.join(a.out, "figures"); os.makedirs(fig_dir, exist_ok=True)
    epochs, patience = (8, 2) if a.quick else (40, 4)
    sel_seeds = list(range(a.select_seeds if not a.quick else 1))
    fin_seeds = list(range(100, 100 + (a.final_seeds if not a.quick else 2)))

    # ---------------------------------------------------------------- data + EDA
    d = D.prepare(a.data, a.seed, 0.2, a.min_count)
    pre = d["pre"]; ytest = d["test"][2]
    eda = compute_eda(d)
    plots.plot_eda(eda, os.path.join(fig_dir, "01_eda.png"))
    print(f"[data] train/val/test = {len(d['train'][2])}/{len(d['val'][2])}/{len(ytest)}  "
          f"click rate {eda['positive_rate']['train']:.3%}  dense dims {pre.n_dense}  "
          f"embedding rows {eda['total_embedding_rows']:,}")

    # ---------------------------------------------------------------- architecture search (validation only)
    g = (lambda grid: S.quick(grid)) if a.quick else (lambda grid: grid)
    families = [("logreg", g(S.LOGREG_GRID)), ("fm", g(S.FM_GRID)), ("mlp", g(S.MLP_GRID)), ("dcn", g(S.DCN_GRID))]
    tables, best_specs, curve_runs = [], {}, []
    for fam, grid in families:
        print(f"[select] {fam}: {len(grid)} configs x {len(sel_seeds)} seed(s)")
        df, best, runs0 = select(grid, d, sel_seeds, epochs, patience, family=fam)
        tables.append(df); best_specs[fam] = best
        if fam in ("mlp", "dcn"):
            curve_runs += runs0
    grid_df = pd.concat(tables, ignore_index=True)
    grid_df.to_csv(os.path.join(a.out, "grid.csv"), index=False)
    plots.plot_curves_grid(curve_runs, os.path.join(fig_dir, "02_train_val_curves.png"))
    with open(os.path.join(a.out, "selected.json"), "w") as f:
        json.dump({"split_seed": a.seed, "min_count": a.min_count, "specs": best_specs}, f, indent=2)

    # ---------------------------------------------------------------- final evaluation (test touched once per model)
    rows, preds = [], {}
    labels = {"logreg": "LogReg (baseline)", "fm": "FM (MF-style baseline)", "mlp": "Vanilla MLP", "dcn": "DCN (bonus)"}
    for fam, _ in families:
        spec = best_specs[fam]
        print(f"[final] {labels[fam]}: {S.spec_name(spec)}  x {len(fin_seeds)} seeds")
        runs = final_eval(spec, d, fin_seeds, epochs, patience)
        row = summarize(labels[fam], runs, ytest); row["config"] = S.spec_name(spec)
        rows.append(row); preds[labels[fam]] = (ytest, runs[0]["p_test"])
        print(f"   test ROC-AUC {pm(row,'roc_auc')}  PR-AUC {pm(row,'pr_auc')}  logloss {pm(row,'log_loss')}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "final_test_metrics.csv"), index=False)
    plots.plot_bars(res, os.path.join(fig_dir, "03_test_metrics.png"))
    plots.plot_reliability(preds, os.path.join(fig_dir, "04_calibration.png"))
    plots.plot_roc_pr(preds, os.path.join(fig_dir, "05_roc_pr.png"))

    env = {"command": " ".join(sys.argv), "split_seed": a.seed, "selection_seeds": sel_seeds, "final_seeds": fin_seeds,
           "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
           "matplotlib": matplotlib.__version__, "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
           "runtime_s": round(time.time() - t_all, 1), "framework": "NumPy (manual backprop), float32"}
    json.dump(env, open(os.path.join(a.out, "environment.json"), "w"), indent=2)

    # ---------------------------------------------------------------- report
    gcols = ["name", "family", "params", "best_epoch", "train_loss@best", "val_log_loss", "gap@best", "val_roc_auc",
             "val_pr_auc", "selected"]
    def fmt_grid(df):
        df = df[gcols].copy()
        df["best_epoch"] = df["best_epoch"].map(lambda v: f"{v:.1f}")
        df["params"] = df["params"].map(lambda v: f"{int(v):,}")
        df["selected"] = df["selected"].map(lambda v: "**yes**" if v else "")
        return df.drop(columns="family")
    gcols_all = gcols
    mlp_t = fmt_grid(grid_df[grid_df.family == "mlp"]); dcn_t = fmt_grid(grid_df[grid_df.family == "dcn"])
    fin = pd.DataFrame({
        "model": res.model, "config": res.config,
        "ROC-AUC": [pm(r, "roc_auc") for _, r in res.iterrows()],
        "PR-AUC": [pm(r, "pr_auc") for _, r in res.iterrows()],
        "log loss": [pm(r, "log_loss") for _, r in res.iterrows()],
        "ECE": [pm(r, "ece") for _, r in res.iterrows()],
        "F1@thr": [pm(r, "f1") for _, r in res.iterrows()],
        "precision@thr": [pm(r, "precision", 3) for _, r in res.iterrows()],
        "recall@thr": [pm(r, "recall", 3) for _, r in res.iterrows()],
        "acc@thr": [pm(r, "accuracy", 3) for _, r in res.iterrows()],
        "acc@0.5": [f"{r['accuracy@0.5']:.3f}" for _, r in res.iterrows()],
    })
    ci = pd.DataFrame({"model": res.model,
                       "ROC-AUC 95% CI (bootstrap over test rows)": [f"[{r['roc_auc_ci'][0]:.3f}, {r['roc_auc_ci'][1]:.3f}]" for _, r in res.iterrows()],
                       "PR-AUC 95% CI": [f"[{r['pr_auc_ci'][0]:.3f}, {r['pr_auc_ci'][1]:.3f}]" for _, r in res.iterrows()]})
    m, c = res[res.model == "Vanilla MLP"].iloc[0], res[res.model == "DCN (bonus)"].iloc[0]
    dAUC = c.roc_auc - m.roc_auc; noise = max(m.roc_auc_std, c.roc_auc_std)
    verdict = ("larger than" if abs(dAUC) > 2 * noise else "comparable to") 
    unreg = grid_df[(grid_df.family == "mlp")].sort_values("gap@best", ascending=False).iloc[0]
    report = f"""# Task 2 results: neural CTR prediction

Auto-generated by `run_task2.py` ({env['runtime_s']} s, split seed {a.seed}). All models are implemented from scratch in NumPy
(manual back-propagation, gradient-checked in `tests/`), trained with Adam on binary cross-entropy.

## 1. Data and preprocessing decisions
- Supplied files: `train.csv` ({eda['n_rows']['train'] + eda['n_rows']['val']:,} rows) and `test.csv` ({eda['n_rows']['test']:,} rows, **untouched**: never used for fitting, tuning, early stopping or threshold choice).
- Train file split into train/validation = {eda['n_rows']['train']:,}/{eda['n_rows']['val']:,} (stratified by label, seed {a.seed}); the same split is reused by Task 3.
- Click rate: train {eda['positive_rate']['train']:.2%}, val {eda['positive_rate']['val']:.2%}, test {eda['positive_rate']['test']:.2%}. Predicting
  "no click" gives accuracy {eda['majority_class_accuracy_test']:.3f} (so **accuracy is nearly meaningless**) and a prior log loss of {eda['prior_log_loss_test']:.4f}.
- Numeric (13): median imputation + a missing-indicator for the {len(pre.miss_cols)} columns with missing values, signed `log1p` (heavy tails), standardisation; **all statistics fitted on the training part only** -> {pre.n_dense} dense inputs.
- Categorical (26): missing is its own token; categories seen fewer than {a.min_count} times in the training part (and every category unseen in training) share one OOV id 0. {eda['total_embedding_rows']:,} embedding rows in total (raw cardinalities up to {max(eda['cardinality_raw'].values()):,}). OOV rate on test averages {100*np.mean(list(eda['oov_rate_test'].values())):.1f}% per field.
- The OOV threshold (3 / 10 / 30) was compared on validation with LogReg, MLP and DLRM; 10 was best or tied for all three, and it also shrinks the embedding tables ~3x.
- Imbalance handling: **no re-weighting / resampling**, so probabilities stay calibrated; instead the decision threshold is chosen on validation (max F1) and frozen.

![eda](figures/01_eda.png)

## 2. Architecture search (validation only; mean of {len(sel_seeds)} seed(s); selection = lowest validation log loss)
Vanilla MLP = concat[dense, flattened per-field embeddings] -> ReLU hidden layers (+dropout / L2) -> logit. Adam, batch 512, lr 3e-4, early stopping (patience {patience}) on validation log loss, best epoch restored.
`gap@best` = validation minus training log loss at the selected epoch (generalisation gap).

{md_table(mlp_t)}

### Overfitting investigation
![curves](figures/02_train_val_curves.png)

With only {eda['n_rows']['train']:,} training rows, ~{int(round(eda['positive_rate']['train']*eda['n_rows']['train']))} clicks and ~{eda['total_embedding_rows']:,} embedding rows, every network memorises the training set within a few epochs: training loss keeps falling while validation loss turns upward (dotted line = selected epoch). The largest gap among the MLPs is {unreg['gap@best']:.3f} ({unreg['name']}). Remedies tried: smaller embeddings / narrower layers, dropout, weight decay, lower learning rate and early stopping. Strong sparse L2 on embeddings was tried during development and hurt validation loss (it erased signal from rare categories), so the final grid uses dropout + weight decay + early stopping.

## 3. Bonus: Deep & Cross Network (explicit crosses) vs. vanilla MLP
Cross layer: x_(l+1) = x0 * (x_l . w_l) + b_l + x_l, run in parallel with a deep tower.

{md_table(dcn_t)}

## 4. Final test results (best configuration per family, {len(fin_seeds)} seeds, mean ± std)
Threshold = max-F1 on validation, then frozen. Baselines: **LogReg** (linear) and **FM** (factorization machine = Task-1 matrix factorization generalised to all fields).

{md_table(fin)}

{md_table(ci)}

![bars](figures/03_test_metrics.png)
![roc](figures/05_roc_pr.png)

### Calibration
![cal](figures/04_calibration.png)

### Conclusions
- DCN vs vanilla MLP: ROC-AUC difference {dAUC:+.4f}, which is {verdict} the seed-to-seed standard deviation ({noise:.4f}); see the bootstrap intervals above, which are wide (only ~{int(round(eda['positive_rate']['test']*eda['n_rows']['test']))} clicks in test). Explicit crosses are therefore {'a measurable gain' if dAUC > 2*noise else 'not clearly better than a regularised MLP on this small dataset'}.
- Log loss improvements over the prior ({eda['prior_log_loss_test']:.4f}) are small in absolute terms: click prediction on this data is intrinsically noisy, so rank metrics (ROC-AUC, PR-AUC) and calibration matter more than accuracy.
- Validation-set selection is itself noisy (~{int(round(eda['positive_rate']['val']*eda['n_rows']['val']))} validation clicks), which is why 2 seeds are averaged during selection and 5 seeds are reported at the end.

## 5. Environment
{json.dumps({k: env[k] for k in ('python','numpy','pandas','matplotlib','platform','cpu_count','framework')})}; re-run: `{env['command']}`.
"""
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write(report)
    print(f"[done] {env['runtime_s']}s -> {os.path.join(a.out, 'report.md')}")


if __name__ == "__main__":
    main()
