#!/usr/bin/env python
"""Task 1 -- Collaborative filtering: memory-based kNN vs. matrix factorization (from scratch).

Usage:
    python run_experiment.py                  # full run (MovieLens-100K, downloads automatically)
    python run_experiment.py --quick          # tiny grids / few epochs, ~1 minute smoke run
    python run_experiment.py --dataset synthetic
Outputs go to results/ (metrics.csv, report.md, environment.json, example_recommendations.txt, figures/).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time

import matplotlib
import numpy as np
import pandas as pd

from cf import plots
from cf.data import load_dataset, per_user_split, to_matrices
from cf.memory_cf import KNNCF
from cf.metrics import mae, ranking_metrics, rated_ndcg, rmse
from cf.mf import BiasedMF

HERE = os.path.dirname(os.path.abspath(__file__))
RELEVANT_THRESHOLD = 4.0     # held-out rating >= 4 counts as "relevant" for ranking metrics
KS = (5, 10, 20)


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def bucket_rmse(users, items, y, pred, counts, edges, by):
    key = counts[users] if by == "user" else counts[items]
    labels, vals = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (key >= lo) & (key < hi)
        labels.append(f"{lo}-{hi - 1}" if np.isfinite(hi) else f"{lo}+")
        vals.append(rmse(y[m], pred[m]) if m.sum() >= 20 else np.nan)
    return labels, vals


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # never crash on a legacy Windows console encoding
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["auto", "movielens", "synthetic"], default="auto")
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="small grids and few epochs (smoke test)")
    args = ap.parse_args()

    t_start = time.time()
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    np.random.seed(args.seed)

    # ------------------------------------------------------------------ 1. data + EDA
    ds = load_dataset(args.dataset, args.data_dir, args.seed)
    train, val, test = per_user_split(ds.ratings, args.seed)
    U, I = ds.n_users, ds.n_items
    R, M = to_matrices(train, U, I)
    Rv, Mv = to_matrices(val, U, I)
    Rt, Mt = to_matrices(test, U, I)
    print(f"[data] {ds.name}: {U} users, {I} items, {len(ds.ratings)} ratings | "
          f"train/val/test = {len(train)}/{len(val)}/{len(test)}")
    sparsity = 1 - len(ds.ratings) / (U * I)
    train_user_cnt = np.bincount(train.user, minlength=U)
    train_item_cnt = np.bincount(train.item, minlength=I)
    eda = {
        "n_users": U, "n_items": I, "n_ratings": len(ds.ratings), "sparsity": sparsity,
        "rating_mean": float(ds.ratings.rating.mean()), "rating_std": float(ds.ratings.rating.std()),
        "missing_values": int(ds.ratings[["user", "item", "rating"]].isna().sum().sum()),
        "items_without_train_rating": int((train_item_cnt == 0).sum()),
        "ratings_per_user_median": float(np.median(np.bincount(ds.ratings.user, minlength=U))),
        "ratings_per_item_median": float(np.median(np.bincount(ds.ratings.item, minlength=I))),
        "rating_distribution": {int(k): int(v) for k, v in train.rating.value_counts().sort_index().items()},
    }
    plots.plot_eda(train, U, I, os.path.join(fig_dir, "01_eda.png"), title=ds.name)

    tu, ti, tr_ = train.user.values, train.item.values, train.rating.values
    vu, vi, vr = val.user.values, val.item.values, val.rating.values
    su, si, sr = test.user.values, test.item.values, test.rating.values

    # ------------------------------------------------------------------ 2. tune on VALIDATION only
    if args.quick:
        knn_ks, knn_shrinks = [20, 50], [25]
        mf_factors, mf_regs, mf_epochs = [16], [0.05], 15
    else:
        knn_ks, knn_shrinks = [10, 25, 50, 100, 200, 400], [0, 25, 100]
        mf_factors, mf_regs, mf_epochs = [16, 32, 64], [0.02, 0.05, 0.1], 80

    print("[tune] kNN ...")
    knn_tuning, knn_best = {}, {}
    for mode in ("item", "user"):
        best = (np.inf, None)
        for shrink in knn_shrinks:
            pts = []
            for k in knn_ks:
                m = KNNCF(mode, k, shrink).fit(R, M)
                v = rmse(vr, m.predict_pairs(vu, vi))
                pts.append((k, v))
                if v < best[0]:
                    best = (v, (k, shrink))
            knn_tuning[(mode, shrink)] = pts
        knn_best[mode] = best[1]
        print(f"   {mode}-{mode}: best k={best[1][0]}, shrink={best[1][1]}  val RMSE={best[0]:.4f}")
    plots.plot_knn_tuning(knn_tuning, os.path.join(fig_dir, "02_knn_tuning.png"))

    print("[tune] matrix factorization (Adam) ...")
    mf_grid, best_mf = [], (np.inf, None, None)
    for nf, reg in itertools.product(mf_factors, mf_regs):
        m = BiasedMF(U, I, nf, lr=0.01, reg=reg, epochs=mf_epochs, seed=args.seed)
        t0 = time.time()
        m.fit(tu, ti, tr_, val=(vu, vi, vr))
        m.fit_time = time.time() - t0
        v = min(m.history["val_rmse"])
        mf_grid.append({"factors": nf, "reg": reg, "best_epoch": m.best_epoch, "val_rmse": v})
        print(f"   factors={nf:3d} reg={reg:<5} best_epoch={m.best_epoch:3d} val RMSE={v:.4f}")
        if v < best_mf[0]:
            best_mf = (v, (nf, reg), m)
    nf_best, reg_best = best_mf[1]
    mf_model = best_mf[2]

    # optimiser ablation at the selected configuration (Adam vs plain SGD)
    sgd_model = BiasedMF(U, I, nf_best, lr=0.003 if not args.quick else 0.003, reg=reg_best,
                         epochs=mf_epochs, optimizer="sgd", seed=args.seed)
    t0 = time.time()
    sgd_model.fit(tu, ti, tr_, val=(vu, vi, vr))
    sgd_model.fit_time = time.time() - t0
    plots.plot_mf_curves({"Adam (lr=0.01)": mf_model.history, "SGD (lr=0.003)": sgd_model.history},
                         os.path.join(fig_dir, "03_mf_curves.png"))

    # ------------------------------------------------------------------ 3. final fit + TEST evaluation (once)
    print("[eval] test set ...")
    exclude = M | Mv                                        # never recommend already-seen items
    relevant = Mt & (Rt >= RELEVANT_THRESHOLD)
    models, rows, test_preds, score_mats = [], [], {}, {}

    def evaluate(name, fit_fn, pred_fn, scores_fn, family):
        t0 = time.time(); obj = fit_fn(); fit_t = getattr(obj, 'fit_time', time.time() - t0)
        t0 = time.time(); p = pred_fn(obj); sc = scores_fn(obj); pred_t = time.time() - t0
        rk = ranking_metrics(sc, exclude, relevant, KS, seed=args.seed)
        row = {"model": name, "family": family, "rmse": rmse(sr, p), "mae": mae(sr, p),
               "ndcg_rated@5": rated_ndcg(sc, Mt, Rt, 5, args.seed), **rk,
               "fit_s": fit_t, "predict_s": pred_t}
        rows.append(row); test_preds[name] = p; score_mats[name] = sc
        print(f"   {name:34s} RMSE {row['rmse']:.4f}  MAE {row['mae']:.4f}  rNDCG@5 {row['ndcg_rated@5']:.4f}  NDCG@10 {row['ndcg@10']:.4f}  "
              f"Recall@10 {row['recall@10']:.4f}  ({fit_t:.1f}s fit)")
        return obj

    gm = float(tr_.mean())
    evaluate("Global mean (random ranking)", lambda: None, lambda _: np.full(len(sr), gm),
             lambda _: np.zeros((U, I)), "baseline")
    evaluate("Popularity (ranking only)", lambda: None, lambda _: np.full(len(sr), gm),
             lambda _: np.tile(train_item_cnt.astype(float), (U, 1)), "baseline")
    bias_m = evaluate("Bias-only (mu+b_u+b_i)",
                      lambda: BiasedMF(U, I, 0, lr=0.01, reg=reg_best, epochs=mf_epochs, seed=args.seed)
                      .fit(tu, ti, tr_, val=(vu, vi, vr)),
                      lambda m: m.predict_pairs(su, si), lambda m: m.predict_all(), "baseline")
    knn_models = {}
    for mode in ("user", "item"):
        k, shrink = knn_best[mode]
        knn_models[mode] = evaluate(f"Memory CF: {mode}-{mode} kNN (k={k})",
                                    lambda mode=mode, k=k, shrink=shrink: KNNCF(mode, k, shrink).fit(R, M),
                                    lambda m: m.predict_pairs(su, si), lambda m: m.predict_all(), "memory-based")
    evaluate(f"Model CF: MF Adam (f={nf_best})", lambda: mf_model,
             lambda m: m.predict_pairs(su, si), lambda m: m.predict_all(), "model-based")
    evaluate(f"Model CF: MF SGD (f={nf_best})", lambda: sgd_model,
             lambda m: m.predict_pairs(su, si), lambda m: m.predict_all(), "model-based")
    # note: the "popularity" model has no rating predictor, so its RMSE/MAE equal the global-mean model's
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    pd.DataFrame(mf_grid).to_csv(os.path.join(args.out, "mf_tuning.csv"), index=False)

    main_rows = [r for r in rows if r["model"] != "Popularity (ranking only)"]
    plots.plot_comparison(main_rows, os.path.join(fig_dir, "04_comparison.png"))

    # ------------------------------------------------------------------ 4. robustness: activity / popularity buckets
    focus = [r["model"] for r in rows if r["family"] != "baseline" or r["model"].startswith("Bias")]
    focus = [n for n in focus if "SGD" not in n]
    ub = {n: bucket_rmse(su, si, sr, test_preds[n], train_user_cnt, [0, 30, 60, 120, np.inf], "user") for n in focus}
    ib = {n: bucket_rmse(su, si, sr, test_preds[n], train_item_cnt, [0, 5, 20, 100, np.inf], "item") for n in focus}
    plots.plot_buckets(ub, ib, os.path.join(fig_dir, "05_buckets.png"))

    # ------------------------------------------------------------------ 5. qualitative example
    u0 = int(np.argmax(train_user_cnt))
    name_of = (lambda j: ds.item_titles[j]) if ds.item_titles else (lambda j: f"item {j}")
    lines = [f"Example user #{u0} ({train_user_cnt[u0]} training ratings)", "", "Highest-rated training items:"]
    liked = np.argsort(-(R[u0] + np.random.default_rng(0).random(I) * 0.01 * M[u0]))[:5]
    lines += [f"  {R[u0, j]:.0f}*  {name_of(j)}" for j in liked if M[u0, j]]
    for n in [r["model"] for r in rows if r["family"] in ("memory-based", "model-based") and "SGD" not in r["model"]]:
        s = score_mats[n][u0].copy(); s[exclude[u0]] = -np.inf
        lines += ["", f"Top-5 recommendations -- {n}:"]
        lines += [f"  {s[j]:.2f}  {name_of(j)}" for j in np.argsort(-s)[:5]]
    with open(os.path.join(args.out, "example_recommendations.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ 6. environment + report
    env = {
        "command": " ".join(sys.argv), "seed": args.seed, "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
        "dataset": ds.name, "synthetic": ds.is_synthetic, "split": "per-user random 80/10/10",
        "relevance_threshold": RELEVANT_THRESHOLD, "knn_best": {k: list(v) for k, v in knn_best.items()},
        "mf_best": {"factors": nf_best, "reg": reg_best, "lr": 0.01, "optimizer": "adam"},
        "total_runtime_s": round(time.time() - t_start, 1), "eda": eda,
    }
    with open(os.path.join(args.out, "environment.json"), "w") as f:
        json.dump(env, f, indent=2, default=str)

    cols = ["model", "rmse", "mae", "ndcg_rated@5", "precision@10", "recall@10", "ndcg@10", "hitrate@10", "coverage@10", "fit_s"]
    table = md_table(res[cols].rename(columns={"fit_s": "fit time (s)"}))
    rated = res[~res.model.str.startswith(("Global", "Popularity"))]
    best_rmse = rated.loc[rated.rmse.idxmin(), "model"]
    best_ndcg = res.loc[res["ndcg@10"].idxmax(), "model"]
    mem = res[res.family == "memory-based"].iloc[:, :].copy()
    mod = res[res.model.str.contains("MF Adam")]
    obs = [f"- Lowest test RMSE: **{best_rmse}**. Highest NDCG@10: **{best_ndcg}**.",
           f"- Best memory-based RMSE {mem.rmse.min():.4f} vs. matrix factorization RMSE {mod.rmse.iloc[0]:.4f} "
           f"(MF is {abs(1 - mod.rmse.iloc[0] / mem.rmse.min()) * 100:.1f}% "
           f"{'lower' if mod.rmse.iloc[0] < mem.rmse.min() else 'higher'}).",
           f"- Bias-only baseline RMSE {res[res.model.str.startswith('Bias')].rmse.iloc[0]:.4f}: "
           "any personalised model should beat this to justify its complexity."]
    warn = ("> **WARNING: synthetic fallback data was used (MovieLens download failed). "
            "Do not report these numbers as real-world results.**\n\n") if ds.is_synthetic else ""
    report = f"""# Task 1 results: memory-based vs. model-based collaborative filtering

{warn}Auto-generated by `run_experiment.py` (seed {args.seed}, {env['total_runtime_s']} s).

## Dataset
{ds.description}

{U} users, {I} items, {len(ds.ratings):,} ratings, sparsity {sparsity:.2%}, mean rating {eda['rating_mean']:.2f},
missing values: {eda['missing_values']}. Median ratings per user {eda['ratings_per_user_median']:.0f}, per item
{eda['ratings_per_item_median']:.0f}. Items with no training rating: {eda['items_without_train_rating']}.

![eda](figures/01_eda.png)

## Protocol
- Split: per-user random 80/10/10 (train {len(train):,} / val {len(val):,} / test {len(test):,}), seed {args.seed}.
- All statistics (means, similarities, factors) are learned from **train only**; hyper-parameters are chosen on
  **validation RMSE**; the test set is evaluated once.
- Rating metrics: RMSE, MAE (predictions clipped to [1,5]).
- Rated-item ordering: NDCG@5 (graded gain 2^r-1) over each user's held-out rated items only -- "does the model order
  the movies this user rated by how much they liked them?" (column `ndcg_rated@5`; ignores exposure).
- Full-catalogue ranking metrics: for each user, rank all items not seen in train/val; relevant = test rating >= {RELEVANT_THRESHOLD:g};
  Precision/Recall/NDCG/HitRate@K and catalogue coverage@K averaged over users with >=1 relevant item
  ({res['n_eval_users'].iloc[0]} users).
- Selected: item-item k={knn_best['item'][0]}, shrink={knn_best['item'][1]}; user-user k={knn_best['user'][0]},
  shrink={knn_best['user'][1]}; MF factors={nf_best}, L2={reg_best}, Adam lr=0.01, batch 1024, early stopping on val.

## Test results
{table}

{chr(10).join(obs)}

Note: full-catalogue top-K is dominated by *exposure* (which items people chose to rate): the popularity baseline is
hard to beat there for models trained to predict rating values, because they favour obscure items with few but high
ratings. `ndcg_rated@5` isolates how well the predicted rating itself orders items.

![comparison](figures/04_comparison.png)

## Tuning and training curves
![knn](figures/02_knn_tuning.png)
![mf](figures/03_mf_curves.png)

MF validation grid: `mf_tuning.csv`.

## Where does each method win? (RMSE by user activity / item popularity)
![buckets](figures/05_buckets.png)

## Environment
Python {env['python']}, numpy {env['numpy']}, pandas {env['pandas']}, matplotlib {env['matplotlib']},
{env['platform']}, {env['cpu_count']} CPU core(s). Re-run: `{env['command']}`.
"""
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[done] {env['total_runtime_s']}s -> {args.out}/report.md")


if __name__ == "__main__":
    main()
