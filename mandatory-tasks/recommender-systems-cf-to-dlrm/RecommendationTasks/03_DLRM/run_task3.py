#!/usr/bin/env python
"""Task 3 -- DLRM from scratch + ablations + final comparison across the whole progression.

    python run_task3.py            # full run (~6-10 min on 1 CPU core). Run Task 2 first (re-uses its selected configs).
    python run_task3.py --quick
Outputs -> results/ : report.md, dlrm_grid.csv, ablations.csv, final_comparison.csv, figures/ ...
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
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from ctrlib import data as D, plots, specs as S            # noqa: E402
from ctrlib.experiment import final_eval, md_table, pm, select, summarize   # noqa: E402
from ctrlib.eda import compute_eda                         # noqa: E402


def load_task2_specs(quick):
    p = os.path.join(ROOT, "02_NeuralCTR", "results", "selected.json")
    if os.path.exists(p):
        j = json.load(open(p))
        print(f"[task2] using the configurations selected in Task 2 ({p})")
        return j["specs"], j.get("split_seed", 42), j.get("min_count", 10), True
    print("[task2] WARNING: Task 2 results not found -> using default configs from ctrlib/specs.py. "
          "Run 02_NeuralCTR/run_task2.py first for a faithful comparison.")
    return ({"logreg": S.LOGREG_GRID[0], "fm": S.FM_GRID[1], "mlp": S.MLP_GRID[3], "dcn": S.DCN_GRID[0]}, 42, 10, False)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # never crash on a legacy Windows console encoding
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--select-seeds", type=int, default=2)
    ap.add_argument("--final-seeds", type=int, default=5)
    ap.add_argument("--ablation-seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t_all = time.time()
    fig_dir = os.path.join(a.out, "figures"); os.makedirs(fig_dir, exist_ok=True)
    epochs, patience = (8, 2) if a.quick else (40, 4)
    sel_seeds = list(range(a.select_seeds if not a.quick else 1))
    fin_seeds = list(range(100, 100 + (a.final_seeds if not a.quick else 2)))     # same seed policy as Task 2
    abl_seeds = fin_seeds[:a.ablation_seeds] if not a.quick else fin_seeds[:2]

    t2_specs, split_seed, min_count, from_t2 = load_task2_specs(a.quick)
    d = D.prepare(a.data, split_seed, 0.2, min_count)
    ytest = d["test"][2]
    eda = compute_eda(d)
    print(f"[data] same split as Task 2 (seed {split_seed}): {len(d['train'][2])}/{len(d['val'][2])}/{len(ytest)}")

    # ---------------------------------------------------------------- DLRM selection on validation
    grid = S.quick(S.DLRM_GRID) if a.quick else S.DLRM_GRID
    print(f"[select] DLRM: {len(grid)} configs x {len(sel_seeds)} seed(s)")
    grid_df, best_dlrm, _ = select(grid, d, sel_seeds, epochs, patience, family="dlrm")
    grid_df.to_csv(os.path.join(a.out, "dlrm_grid.csv"), index=False)

    # ---------------------------------------------------------------- final comparison (same split, seeds, metrics)
    labels = {"logreg": "LogReg", "fm": "FM (MF-style)", "mlp": "Vanilla MLP (Task 2)", "dcn": "DCN (Task 2 bonus)"}
    entries = [(labels[k], t2_specs[k]) for k in ("logreg", "fm", "mlp", "dcn")] + [("DLRM (Task 3)", best_dlrm)]
    rows, preds, dlrm_runs = [], {}, None
    for name, spec in entries:
        print(f"[final] {name}: {S.spec_name(spec)} x {len(fin_seeds)} seeds")
        runs = final_eval(spec, d, fin_seeds, epochs, patience)
        row = summarize(name, runs, ytest); row["config"] = S.spec_name(spec)
        rows.append(row); preds[name] = (ytest, runs[0]["p_test"])
        if name.startswith("DLRM"):
            dlrm_runs = runs
        print(f"   test ROC-AUC {pm(row,'roc_auc')}  PR-AUC {pm(row,'pr_auc')}  logloss {pm(row,'log_loss')}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "final_comparison.csv"), index=False)
    plots.plot_bars(res, os.path.join(fig_dir, "01_final_comparison.png"), "Same split, seeds and metrics: LR -> FM -> MLP -> DCN -> DLRM")
    plots.plot_cost(res, os.path.join(fig_dir, "02_cost.png"))
    plots.plot_roc_pr(preds, os.path.join(fig_dir, "03_roc_pr.png"))
    plots.plot_reliability(preds, os.path.join(fig_dir, "04_calibration.png"))

    # ---------------------------------------------------------------- ablations around the selected DLRM
    variants = [("DLRM full (selected)", best_dlrm)]
    def var(label, **kw):
        s = dict(best_dlrm); s.update(kw); variants.append((label, s))
    var("A1: no interaction (concat embeddings -> top MLP)", interaction="cat")
    for e in (4, 8, 16, 32):
        if e != best_dlrm["emb_dim"]:
            var(f"A2: embedding dim {e}", emb_dim=e)
    var("A3: dense MLP minimal (1 layer)", bottom=[])
    var("A3: dense MLP deeper [128, 64]", bottom=[128, 64])
    var("A4: smaller top MLP [32]", top=[32])
    if a.quick:
        variants = variants[:3]
    abl_rows = []
    for label, spec in variants:
        if label.startswith("DLRM full"):
            runs = dlrm_runs[:len(abl_seeds)]
        else:
            print(f"[ablation] {label}")
            runs = final_eval(spec, d, abl_seeds, epochs, patience)
        r = summarize(label, runs, ytest)
        r["val_log_loss"] = float(np.mean([x["val"]["log_loss"] for x in runs]))
        abl_rows.append(r)
        print(f"   {label:52s} test ROC-AUC {pm(r,'roc_auc')}  logloss {pm(r,'log_loss')}  params {int(r['n_params']):,}")
    abl = pd.DataFrame(abl_rows)
    abl.to_csv(os.path.join(a.out, "ablations.csv"), index=False)
    plots.plot_ablation(abl, os.path.join(fig_dir, "05_ablations.png"))

    # ---------------------------------------------------------------- Task 1 context (different dataset / task)
    t1_path = os.path.join(ROOT, "01_CollaborativeFiltering", "results", "metrics.csv")
    t1_md = ("_Task 1 results not found next to this folder; run `01_CollaborativeFiltering/run_experiment.py` to include them._")
    if os.path.exists(t1_path):
        t1 = pd.read_csv(t1_path)
        t1 = t1[t1.family.isin(["baseline", "memory-based", "model-based"]) & ~t1.model.str.contains("Popularity|SGD")]
        t1 = t1[["model", "rmse", "mae", "ndcg_rated@5"]]
        t1_md = md_table(t1)

    # ---------------------------------------------------------------- report
    env = {"command": " ".join(sys.argv), "split_seed": split_seed, "final_seeds": fin_seeds, "ablation_seeds": abl_seeds,
           "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
           "matplotlib": matplotlib.__version__, "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
           "runtime_s": round(time.time() - t_all, 1), "framework": "NumPy (manual backprop), float32",
           "selected_dlrm": best_dlrm, "task2_configs_from_file": from_t2}
    json.dump(env, open(os.path.join(a.out, "environment.json"), "w"), indent=2, default=str)

    g = lambda n: res[res.model == n].iloc[0]
    dl, ml, dc, fm, lr = g("DLRM (Task 3)"), g("Vanilla MLP (Task 2)"), g("DCN (Task 2 bonus)"), g("FM (MF-style)"), g("LogReg")
    def cmp(a_, b_, nm):
        dA = a_.roc_auc - b_.roc_auc; sd = max(a_.roc_auc_std, b_.roc_auc_std)
        tag = "ahead of" if dA > 2 * sd else ("behind" if dA < -2 * sd else "statistically indistinguishable from")
        return f"- DLRM vs {nm}: ROC-AUC {dA:+.4f}, PR-AUC {a_.pr_auc - b_.pr_auc:+.4f}, log loss {a_.log_loss - b_.log_loss:+.4f} -> **{tag}** (threshold = 2 seed-std = {2*sd:.4f})."
    full = abl[abl.model.str.startswith("DLRM full")].iloc[0]
    no_int = abl[abl.model.str.startswith("A1")].iloc[0]
    emb_share = 100 * dl.embedding_params / dl.n_params

    fin = pd.DataFrame({"model": res.model, "config": res.config,
        "ROC-AUC": [pm(r, "roc_auc") for _, r in res.iterrows()], "PR-AUC": [pm(r, "pr_auc") for _, r in res.iterrows()],
        "log loss": [pm(r, "log_loss") for _, r in res.iterrows()], "ECE": [pm(r, "ece") for _, r in res.iterrows()],
        "F1@thr": [pm(r, "f1") for _, r in res.iterrows()], "precision@thr": [pm(r, "precision", 3) for _, r in res.iterrows()],
        "recall@thr": [pm(r, "recall", 3) for _, r in res.iterrows()], "acc@thr": [pm(r, "accuracy", 3) for _, r in res.iterrows()],
        "acc@0.5": [f"{r['accuracy@0.5']:.3f}" for _, r in res.iterrows()]})
    cost = pd.DataFrame({"model": res.model, "params": [f"{int(v):,}" for v in res.n_params],
        "embedding share": [f"{100*e/n:.0f}%" for e, n in zip(res.embedding_params, res.n_params)],
        "memory fp32 (MB)": [f"{v:.2f}" for v in res.memory_mb_fp32], "s / epoch": [f"{v:.2f}" for v in res.sec_per_epoch],
        "train time (s)": [f"{v:.1f}" for v in res.train_time_s], "best epoch": [f"{v:.1f}" for v in res.best_epoch],
        "inference µs/row": [f"{v:.1f}" for v in res.infer_us_per_row]})
    abl_t = pd.DataFrame({"variant": abl.model, "params": [f"{int(v):,}" for v in abl.n_params],
        "val log loss": [f"{v:.4f}" for v in abl.val_log_loss], "test ROC-AUC": [pm(r, "roc_auc") for _, r in abl.iterrows()],
        "test PR-AUC": [pm(r, "pr_auc") for _, r in abl.iterrows()], "test log loss": [pm(r, "log_loss") for _, r in abl.iterrows()],
        "ΔROC-AUC vs full": [f"{r.roc_auc - full.roc_auc:+.4f}" for _, r in abl.iterrows()]})
    gt = grid_df.copy(); gt["best_epoch"] = gt.best_epoch.map(lambda v: f"{v:.1f}"); gt["params"] = gt.params.map(lambda v: f"{int(v):,}")
    gt["selected"] = gt.selected.map(lambda v: "**yes**" if v else ""); gt = gt[["name", "params", "best_epoch", "train_loss@best", "val_log_loss", "gap@best", "val_roc_auc", "val_pr_auc", "selected"]]
    n_pairs = 27 * 26 // 2
    d_ml = dl.roc_auc - ml.roc_auc; sd_ml = max(dl.roc_auc_std, ml.roc_auc_std)
    n_tr, n_clk = len(d['train'][2]), int(round(eda['positive_rate']['train'] * len(d['train'][2])))
    if d_ml > 2 * sd_ml:
        obs_text = "DLRM improves on the simpler models."
    elif d_ml < -2 * sd_ml:
        obs_text = (f"with only {n_tr} training rows (~{n_clk} clicks) DLRM is **worse** than the regularised vanilla MLP "
                    f"(ROC-AUC {d_ml:+.4f}, beyond 2 seed-std) and not better than the linear baseline: its extra capacity and the "
                    "351 pairwise-interaction features are not rewarded and mainly add over-fitting risk. This is consistent with DLRM "
                    "being designed for very large click logs; paper-scale gains should not be expected from a 40k-row sample.")
    else:
        obs_text = (f"with only {n_tr} training rows (~{n_clk} clicks) DLRM is not clearly better than the regularised MLP/DCN or the "
                    "linear baseline. This is consistent with DLRM being designed for very large click logs.")

    report = f"""# Task 3 results: DLRM and the full progression

Auto-generated by `run_task3.py` ({env['runtime_s']} s). DLRM is implemented from scratch (NumPy, manual back-propagation, gradient-checked in
`tests/`); no pretrained weights and no DLRM library. Same train/validation/test split (seed {split_seed}), same preprocessing, same seed policy
(seeds {fin_seeds[0]}..{fin_seeds[-1]}) and same metrics as Task 2. Paper notes: `PAPER_NOTES.md`.

## 1. Architecture implemented
dense(24) -> **bottom MLP** -> z0 (dim d) ; 26 categorical fields -> **embedding tables** (dim d, {eda['total_embedding_rows']:,} rows total) ;
**interaction** = all pairwise dot products among the 27 vectors {{z0, e_1..e_26}} -> {n_pairs} numbers ; **top MLP** over concat[z0, dots] -> logit -> sigmoid / BCE.
Data handling is leakage-free (see Task 2 report): vocabularies, medians, means, stds from the training part only.

## 2. DLRM hyper-parameter search (validation only, mean of {len(sel_seeds)} seed(s))
{md_table(gt)}

## 3. Final comparison across the progression (test, {len(fin_seeds)} seeds, mean ± std)
LogReg -> **FM** (Task 1's matrix factorization generalised to all fields) -> **vanilla MLP** (Task 2) -> **DCN** (Task 2 bonus) -> **DLRM**.
Threshold = max-F1 on validation, frozen for test. "acc@0.5" shows why accuracy is uninformative here (always-"no click" scores {eda['majority_class_accuracy_test']:.3f}).

{md_table(fin)}

![final](figures/01_final_comparison.png)
![roc](figures/03_roc_pr.png)

{chr(10).join([cmp(dl, ml, 'vanilla MLP'), cmp(dl, dc, 'DCN'), cmp(dl, fm, 'FM (MF-style)'), cmp(dl, lr, 'LogReg')])}

### Calibration
![cal](figures/04_calibration.png)

### Training cost, parameters and memory
{md_table(cost)}

![cost](figures/02_cost.png)

Embeddings hold {emb_share:.0f}% of DLRM's parameters here; with production-size vocabularies (10^7-10^9 rows) that share approaches 100% and
memory, not FLOPs, becomes the constraint (the paper's motivation for model-parallel embedding tables). The interaction layer costs O(F²·d) per example
(F = 27) and widens the top-MLP input to {16 + n_pairs - 16 if False else 'd + ' + str(n_pairs)}.

## 4. Ablations (around the selected DLRM; hyper-parameters not re-tuned per variant; {len(abl_seeds)} seeds)
{md_table(abl_t)}

![abl](figures/05_ablations.png)

- Removing the interaction layer (A1) changes test ROC-AUC by {no_int.roc_auc - full.roc_auc:+.4f} (seed std {max(no_int.roc_auc_std, full.roc_auc_std):.4f}) and log loss by {no_int.log_loss - full.log_loss:+.4f}.
  {'Explicit interactions help here.' if no_int.roc_auc - full.roc_auc < -2 * max(no_int.roc_auc_std, full.roc_auc_std) else 'On this small dataset explicit dot-product interactions do not give a clear benefit.'}

## 5. What does DLRM gain over MF and the vanilla network, and what does it cost?
**Over matrix factorization / FM (Task 1 lineage).** MF/FM models are linear in the embeddings and capture only pairwise "taste x item" similarity through a single dot product per pair; they have no route for numeric context features and no non-linearity. DLRM keeps the same inductive bias (dot products between embeddings) but (i) routes dense features through a bottom MLP into the same latent space, (ii) feeds all pairwise similarities to a non-linear top MLP, so it can learn *which* interactions matter and how they combine, and (iii) supports many heterogeneous fields.
**Over the vanilla MLP (Task 2).** An MLP on concatenated embeddings must discover multiplicative feature interactions from sums of weighted inputs, which is statistically inefficient; DLRM hands it all second-order interactions explicitly (like DCN, but between *learned embeddings* rather than raw concatenated inputs), and the dot product is a strong, parameter-free inductive bias for "affinity".
**Added complexity.** More hyper-parameters (embedding dim shared by all fields and by the bottom-MLP output, two MLPs), quadratic growth of interaction features with the number of fields, embedding tables that dominate memory (and need model-parallel sharding at scale), and an easier route to over-fitting when data is scarce.
**What we actually observed.** See the comparison table: {obs_text} Differences of a few thousandths of ROC-AUC are inside the noise of a test set with ~{int(round(eda['positive_rate']['test']*len(ytest)))} clicks (see bootstrap intervals in `final_comparison.csv`).

## 6. The three tasks side by side
| step | model family | data | what is learned |
|---|---|---|---|
| Task 1a | memory-based CF (user/item kNN) | MovieLens ratings | similarities computed directly from the interaction matrix |
| Task 1b | matrix factorization | MovieLens ratings | user/item latent vectors, score = dot product + biases |
| Task 2 | vanilla MLP / DCN | ad impressions (13 dense + 26 categorical) | non-linear function of concatenated embeddings (+ explicit crosses for DCN) |
| Task 3 | DLRM | same as Task 2 | dense MLP + embeddings + explicit pairwise dot-product interactions + top MLP |

Task 1 results (rating prediction on a different dataset; **not numerically comparable** with the CTR metrics above):

{t1_md}

On the CTR data the MF lineage is represented by the FM row above (same family: biases + dot products of embeddings, generalised from two ID fields to all fields).

## 7. Environment
{json.dumps({k: env[k] for k in ('python','numpy','pandas','matplotlib','platform','cpu_count','framework')})}; re-run: `{env['command']}`.
"""
    open(os.path.join(a.out, "report.md"), "w", encoding="utf-8").write(report)
    print(f"[done] {env['runtime_s']}s -> {os.path.join(a.out, 'report.md')}")


if __name__ == "__main__":
    main()
