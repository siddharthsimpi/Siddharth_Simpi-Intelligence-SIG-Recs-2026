"""Error categorisation (which modality was right/wrong?) and slice-wise error rates."""
import numpy as np
import pandas as pd


def modality_votes(F, text_cols, img_cols, t_text, t_img):
    """Boolean 'this modality alone would predict a match' per pair (thresholds tuned on val per modality)."""
    return F[text_cols].values.mean(1) >= t_text, F[img_cols].values.mean(1) >= t_img


def categorize(F, y, pred, text_cols, img_cols, t_text, t_img):
    tv, iv = modality_votes(F, text_cols, img_cols, t_text, t_img)
    cat = np.full(len(F), "", dtype=object)
    fp, fn = (pred == 1) & (y == 0), (pred == 0) & (y == 1)
    cat[fp & tv & iv] = "FP: text AND image agree (near-identical variant / similar product)"
    cat[fp & tv & ~iv] = "FP: text matches, images differ"
    cat[fp & ~tv & iv] = "FP: image matches, titles differ (visually similar product)"
    cat[fp & ~tv & ~iv] = "FP: fused model over-trusted weak cues"
    cat[fn & tv & ~iv] = "FN: titles match, images differ (different photos of same product)"
    cat[fn & ~tv & iv] = "FN: image matches, titles differ (noisy / disjoint titles)"
    cat[fn & ~tv & ~iv] = "FN: neither modality matches (hard)"
    cat[fn & tv & iv] = "FN: both modalities match but fused model rejected (threshold / calibration)"
    return cat


def slice_rates(F, y, pred):
    """Error rate for interpretable slices -> turns anecdotes into evidence."""
    pos, neg = y == 1, y == 0
    out = []

    def add(name, mask, kind):
        m = mask & (pos if kind == "FN" else neg)
        n = int(m.sum())
        if n:
            err = float(((pred == 0) if kind == "FN" else (pred == 1))[m].mean())
            out.append({"slice": name, "type": kind, "n_pairs": n, "error_rate": err})

    add("all true matches", np.ones(len(F), bool), "FN")
    add("true matches with a very short title (<=2 words)", F["min_words"].values <= 2, "FN")
    add("true matches with very different images (pHash Ham >= 24)", F["ham"].values >= 24, "FN")
    add("true matches with near-identical images (pHash Ham <= 4)", F["ham"].values <= 4, "FN")
    add("true matches whose number tokens conflict", F["num_conflict"].values == 1, "FN")
    add("all true non-matches", np.ones(len(F), bool), "FP")
    add("non-matches with identical image hash", F["phash_eq"].values == 1, "FP")
    add("non-matches whose number tokens conflict (variants)", F["num_conflict"].values == 1, "FP")
    add("non-matches with high title word overlap (Jaccard>=0.6)", F["word_jacc"].values >= 0.6, "FP")
    return pd.DataFrame(out)


def missed_by_retrieval(ctx, split="test", max_show=2000):
    """True matches that never made it into the candidate pool (the recall ceiling of stage 1)."""
    lab, rows, N = ctx.df["label_group"].values, ctx.rows[split], ctx.N
    c = ctx.cand[split]
    have = set((c.i.values.astype(np.int64) * N + c.j.values).tolist())
    s = pd.Series(rows).groupby(lab[rows]).apply(list)
    miss = []
    for g in s:
        for a in g:
            for b in g:
                if a != b and (a * N + b) not in have:
                    miss.append((a, b))
    total = int(sum(len(g) * (len(g) - 1) for g in s))
    return np.array(miss[:max_show]).reshape(-1, 2), len(miss), total
