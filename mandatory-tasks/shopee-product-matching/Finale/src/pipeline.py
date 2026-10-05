"""Evaluation helpers shared by all Finale experiments (thresholds tuned on val, reported on test)."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score

from shopee_match.metrics import best_retrieval_threshold, candidate_recall, retrieval_f1


class Ctx:
    """Everything needed to score a method: candidate tables, features and split bookkeeping."""

    def __init__(self, df, rows, gs, cand, feats):
        self.df, self.rows, self.gs, self.N = df, rows, gs, len(df)
        self.cand, self.feats = cand, feats          # dicts keyed by split name


def tune(ctx, score_val, split="val", n_grid=80):
    c = ctx.cand[split]
    return best_retrieval_threshold(c.i.values, c.j.values, score_val, c.y.values, ctx.rows[split], ctx.gs, ctx.N, n_grid)


def at_threshold(ctx, score, thr, split):
    c = ctx.cand[split]
    return retrieval_f1(c.i.values, c.j.values, score, c.y.values, thr, ctx.rows[split], ctx.gs, ctx.N)


def evaluate(ctx, name, score_val, score_test, **extra):
    """Pick the threshold on VAL, report on TEST."""
    thr, f1_val, _ = tune(ctx, score_val)
    r = at_threshold(ctx, score_test, thr, "test")
    ct = ctx.cand["test"]
    out = {"method": name, "thr": thr, "val_F1": f1_val, "test_F1": r["f1"], "test_P": r["precision"], "test_R": r["recall"],
           "pair_AP": float(average_precision_score(ct.y.values, score_test))}
    out.update(extra)
    return out


def cand_recall(ctx, split="test"):
    c = ctx.cand[split]
    return candidate_recall(c.i.values, c.y.values, ctx.rows[split], ctx.gs, ctx.N)


# ------------------------------------------------------------------ graph post-processing
def symmetrize(ctx, score, thr, split):
    """If i lists j as a match, also let j list i (top-k lists are not symmetric)."""
    c, lab = ctx.cand[split], ctx.df["label_group"].values
    keep = score >= thr
    I = np.concatenate([c.i.values[keep], c.j.values[keep]]); J = np.concatenate([c.j.values[keep], c.i.values[keep]])
    key = np.unique(I.astype(np.int64) * ctx.N + J)
    I, J = key // ctx.N, key % ctx.N
    Y = (lab[I] == lab[J]).astype(np.int8)
    return retrieval_f1(I, J, np.ones(len(I)), Y, 0.5, ctx.rows[split], ctx.gs, ctx.N)


def components(ctx, score, thr, split, max_size=60):
    """Transitive closure: connected components of the 'match' graph. Components larger than max_size are
    discarded (replaced by singletons) to stop a single wrong edge from merging two big products."""
    c, lab = ctx.cand[split], ctx.df["label_group"].values
    keep = score >= thr
    g = coo_matrix((np.ones(keep.sum()), (c.i.values[keep], c.j.values[keep])), shape=(ctx.N, ctx.N))
    _, comp = connected_components(g, directed=False)
    rows = ctx.rows[split]
    comp_r = comp[rows]
    size = pd.Series(comp_r).map(pd.Series(comp_r).value_counts()).values
    comp_r = np.where(size > max_size, -(rows + 1), comp_r)           # oversized -> singleton
    size = pd.Series(comp_r).map(pd.Series(comp_r).value_counts()).values
    cell = pd.DataFrame({"c": comp_r, "l": lab[rows]})
    tp = cell.groupby(["c", "l"])["c"].transform("size").values
    ntrue = ctx.gs[rows]
    return dict(f1=float(np.mean(2 * tp / (size + ntrue))), precision=float(np.mean(tp / size)), recall=float(np.mean(tp / ntrue)))


# ------------------------------------------------------------------ output
def build_match_lists(rows, I, J, mode, N, max_size=60):
    """Turn predicted match edges into {row -> [rows]} (always includes itself).
    mode: 'none' (as predicted) | 'symmetric' (add reverse edges) | 'components' (transitive closure)."""
    rows = np.asarray(rows)
    m = {int(r): [int(r)] for r in rows}
    I, J = np.asarray(I, np.int64), np.asarray(J, np.int64)
    if mode == "components":
        g = coo_matrix((np.ones(len(I)), (I, J)), shape=(N, N))
        _, comp = connected_components(g, directed=False)
        for members in pd.Series(rows).groupby(comp[rows]).apply(list):
            if len(members) <= max_size:
                for r in members:
                    m[int(r)] = [int(x) for x in members]
        return m
    if mode == "symmetric":
        I, J = np.concatenate([I, J]), np.concatenate([J, I])
        key = np.unique(I * N + J)
        I, J = key // N, key % N
    for a, b in zip(I, J):
        m[int(a)].append(int(b))
    return m


def predicted_matches(ctx, score, thr, split, mode="none"):
    """Kaggle-style table: posting_id, matches (space separated)."""
    c, df = ctx.cand[split], ctx.df
    keep = score >= thr
    m = build_match_lists(ctx.rows[split], c.i.values[keep], c.j.values[keep], mode, ctx.N)
    pid = df["posting_id"].values
    return pd.DataFrame({"posting_id": [pid[r] for r in m], "matches": [" ".join(pid[x] for x in v) for v in m.values()]})
