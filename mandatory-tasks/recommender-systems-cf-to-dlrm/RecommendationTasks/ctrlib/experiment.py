"""Selection / final-evaluation helpers shared by the Task 2 and Task 3 scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import average_precision, log_loss, roc_auc
from .specs import spec_name
from .trainer import run_one

METRICS = ["roc_auc", "pr_auc", "log_loss", "ece", "f1", "precision", "recall", "accuracy"]


def select(specs, data, seeds, epochs=25, patience=4, family=None, verbose=True):
    """Train every spec for every seed; rank by mean VALIDATION log loss (lower = better)."""
    rows, runs0 = [], []
    for spec in specs:
        rs = [run_one(spec, s, data, epochs=epochs, patience=patience) for s in seeds]
        h0 = rs[0]["history"]
        best_i = h0["best_epoch"] - 1
        row = {
            "name": spec_name(spec), "family": family or spec["type"], "params": rs[0]["n_params"],
            "best_epoch": float(np.mean([r["history"]["best_epoch"] for r in rs])),
            "train_loss@best": float(np.mean([r["history"]["train_loss"][r["history"]["best_epoch"] - 1] for r in rs])),
            "val_log_loss": float(np.mean([r["val"]["log_loss"] for r in rs])),
            "val_roc_auc": float(np.mean([r["val"]["roc_auc"] for r in rs])),
            "val_pr_auc": float(np.mean([r["val"]["pr_auc"] for r in rs])),
            "sec_per_epoch": float(np.mean([r["sec_per_epoch"] for r in rs])),
        }
        row["gap@best"] = row["val_log_loss"] - row["train_loss@best"]
        rows.append(row)
        runs0.append({"spec": spec, "name": row["name"], "history": h0})
        if verbose:
            print(f"   {row['name']:58s} val LL {row['val_log_loss']:.4f}  AUC {row['val_roc_auc']:.4f}  "
                  f"AP {row['val_pr_auc']:.4f}  ep {row['best_epoch']:.1f}  ({row['params']:,} params)")
    df = pd.DataFrame(rows)
    best = int(df.val_log_loss.idxmin())
    df["selected"] = [i == best for i in range(len(df))]
    return df, specs[best], runs0


def final_eval(spec, data, seeds, epochs=25, patience=4):
    return [run_one(spec, s, data, epochs=epochs, patience=patience) for s in seeds]


def bootstrap_ci(y, p, fn, n=300, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(name, runs, y_test, which="test"):
    """Mean +- std over seeds of every metric + cost columns + bootstrap CIs (seed-0 predictions)."""
    out = {"model": name, "n_seeds": len(runs)}
    for m in METRICS:
        v = [r[which][m] for r in runs]
        out[m] = float(np.mean(v)); out[m + "_std"] = float(np.std(v))
    out["threshold"] = float(np.mean([r["threshold"] for r in runs]))
    out["accuracy@0.5"] = float(np.mean([((r["p_test"] >= 0.5) == (y_test == 1)).mean() for r in runs]))
    p0 = runs[0]["p_test"]
    out["roc_auc_ci"] = bootstrap_ci(y_test, p0, roc_auc)
    out["pr_auc_ci"] = bootstrap_ci(y_test, p0, average_precision)
    for c in ("n_params", "embedding_params", "dense_params", "memory_mb_fp32", "train_time_s",
              "sec_per_epoch", "infer_us_per_row"):
        out[c] = float(np.mean([r[c] for r in runs]))
    out["best_epoch"] = float(np.mean([r["history"]["best_epoch"] for r in runs]))
    return out


def pm(row, key, nd=4):
    return f"{row[key]:.{nd}f} ± {row[key + '_std']:.{nd}f}"


def md_table(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
