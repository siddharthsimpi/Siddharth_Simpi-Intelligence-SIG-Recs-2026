# %% [markdown]
# # Part B — Text-Based Product Matching
#
# **Pipeline:** `title → normalise → representation → similarity → threshold → match / no-match`
#
# We (1) set up a leakage-free evaluation, (2) build a TF-IDF baseline, (3) run experiments with
# char n-grams, multilingual sentence embeddings and a hybrid, (4) analyse the decision threshold,
# (5) study errors, (6) conclude.
#
# **Protocol (important for honest numbers):** the data is split *by product group* into train/val/test.
# Every threshold is selected on **val**; every number in the tables is computed on **test**.
# No model here is trained on labels, so `train` is unused in Part B (it is used in the Finale).
#
# **External resources:** scikit-learn `TfidfVectorizer`; `sentence-transformers`
# (`paraphrase-multilingual-MiniLM-L12-v2`, Reimers & Gurevych 2019). The Kaggle-style F1 metric follows the
# competition description.

# %%
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, scipy.sparse as sp
import matplotlib.pyplot as plt

from shopee_match import config as C
from shopee_match.data import load_dataset, split_rows, group_sizes, get_tag
from shopee_match.text import normalize_title, numbers, jaccard
from shopee_match.embeddings import get_embedding, embedding_meta
from shopee_match.retrieval import pair_scores, topk_pairs
from shopee_match.pairs import get_benchmark
from shopee_match.evaluate import evaluate_embedding
from shopee_match.metrics import (auc_ap, best_f1_threshold, pair_metrics, threshold_table,
                                  retrieval_f1, candidate_recall)
from shopee_match.viz import md_table

RES = C.results_dir("PartB")
df = load_dataset(); tag = get_tag(df); rows = split_rows(df); gs = group_sizes(df)
lab, N = df.label_group.values, len(df)
bench = get_benchmark(df, rows, tag)
print({k: len(v) for k, v in rows.items()}, "| device-independent text part")

# %% [markdown]
# ## 1. Data preparation
# * **Normalisation** (`normalize_title`): Unicode NFKC → lowercase → punctuation/emoji → space → collapse spaces.
#   Rationale: removes cosmetic variation (caps, symbols) that carries no product identity. We keep digits
#   (sizes/models) and non-Latin letters (multilingual).
# * **Benchmark pairs** (fixed for Parts B/C/Finale): positives = same `label_group`; negatives = 1/3 random,
#   1/3 *text-hard* (nearest title, different product), 1/3 *image-hard* (nearest phash, different product).
#   Reporting AUC per negative type shows *where* a method breaks.

# %%
ex = df.sample(5, random_state=3)[["title"]].copy(); ex["normalised"] = ex.title.map(normalize_title); print(ex.to_string())
print("\nbenchmark composition (test):", bench["test"].kind.value_counts().to_dict())

# %% [markdown]
# ## 2. Metrics
# * **Pair level** (is this pair a match?): ROC-AUC (threshold-free), Average Precision, and
#   precision / recall / F1 at the val-selected threshold.
# * **Retrieval level** (Kaggle metric): for each listing predict `{itself} ∪ {top-50 neighbours with score ≥ thr}`;
#   per-row F1 = 2·TP / (|pred| + |true group|); averaged over listings. This is stricter than pair F1 because
#   *every* listing counts equally and the candidate pool is the whole split, not a curated benchmark.
#   `cand_recall` is the maximum recall reachable with top-50 retrieval.

# %% [markdown]
# ## 3. Experiments

# %%
results, E = [], {}
def run(exp, name, rep, sim, metric="cos", note=""):
    Ei = E[name] if name in E else get_embedding(df, name, tag, normalize=(metric != "dot_raw" and metric != "l2_raw"))
    E[name] = Ei
    m = {"dot_raw": "dot", "l2_raw": "l2"}.get(metric, metric)
    r = evaluate_embedding(Ei, df, rows, bench, gs, metric=m)
    meta = embedding_meta(tag, name)
    results.append({"Experiment": exp, "Representation": rep, "Similarity": sim, "key": name, "metric": m, "dim": r["dim"],
                    "embed_s": meta.get("seconds", np.nan), "pair_thr": r["pair_thr"], "AUC": r["pair_auc"], "AP": r["pair_ap"],
                    "pair_P": r["pair_precision"], "pair_R": r["pair_recall"], "pair_F1": r["pair_f1"],
                    "retr_thr": r["retr_thr"], "retr_F1": r["retr_f1"], "retr_P": r["retr_precision"], "retr_R": r["retr_recall"],
                    "cand_R": r["cand_recall"], **{f"AUC_{k}": v for k, v in r["auc_by_negative"].items()}, "note": note})
    print(f"{exp:12s} {rep:28s} AUC={r['pair_auc']:.3f} pairF1={r['pair_f1']:.3f} retrF1={r['retr_f1']:.3f}")

run("Baseline", "tfidf_word", "TF-IDF word 1-gram", "cosine")

# %% [markdown]
# **Baseline:** TF-IDF (sublinear tf, L2-norm) over word unigrams + cosine. *Why:* cheap, no training,
# strong for exact keyword overlap (brand/model words), and fully interpretable. Weakness: any spelling
# variation / synonym / language switch is a different, orthogonal dimension.

# %% [markdown]
# ### Experiment 1 — word n-grams (1–2)
# *Hypothesis:* bigrams capture phrases ("face wash", "iphone 11") and reduce false matches from shared single words.

# %%
run("Exp 1a", "tfidf_word12", "TF-IDF word 1-2gram", "cosine")

# %% [markdown]
# ### Experiment 2 — character n-grams (2–5, `char_wb`)
# *Hypothesis:* char n-grams are robust to typos, abbreviations, concatenated tokens ("500ml" vs "500 ml"),
# morphology and mixed languages, so recall should improve over words.

# %%
run("Exp 2", "tfidf_char", "TF-IDF char_wb 2-5", "cosine")

# %% [markdown]
# ### Experiment 3 — multilingual sentence embeddings
# *Hypothesis:* a pretrained multilingual transformer maps paraphrases / synonyms / different languages to nearby
# vectors, fixing false negatives that keyword methods miss — but may *over-generalise* variants
# (hurting precision on hard negatives).

# %%
run("Exp 3", C.SBERT, "SBERT multilingual MiniLM", "cosine")

# %% [markdown]
# ### Experiment 4 — which similarity measure? (same SBERT vectors)
# Cosine ignores vector length; raw dot product and Euclidean distance do not. We test whether normalisation matters.

# %%
E["sbert_raw"] = get_embedding(df, C.SBERT, tag, normalize=False)
run("Exp 4a", "sbert_raw", "SBERT (raw, un-normalised)", "dot product", metric="dot_raw")
run("Exp 4b", "sbert_raw", "SBERT (raw, un-normalised)", "euclidean (neg.)", metric="l2_raw")

# %% [markdown]
# ### Experiment 5 — hybrid (lexical + semantic)
# *Hypothesis:* char TF-IDF (precise on tokens) and SBERT (semantic) make different errors; averaging their
# cosine scores should beat both. Implemented as cosine of the concatenation `[√.5·x_tfidf, √.5·x_sbert]`,
# which equals the mean of the two cosines.

# %%
Eh = sp.hstack([np.sqrt(.5) * E["tfidf_char"] if "tfidf_char" in E else np.sqrt(.5) * get_embedding(df, "tfidf_char", tag),
                sp.csr_matrix(np.sqrt(.5) * get_embedding(df, C.SBERT, tag))]).tocsr()
E["hybrid"] = Eh
results_before = len(results)
r = evaluate_embedding(Eh, df, rows, bench, gs)
results.append({"Experiment": "Exp 5", "Representation": "char TF-IDF ⊕ SBERT", "Similarity": "cosine (mean of both)", "key": "hybrid", "metric": "cos",
                "dim": r["dim"], "embed_s": np.nan, "pair_thr": r["pair_thr"], "AUC": r["pair_auc"], "AP": r["pair_ap"],
                "pair_P": r["pair_precision"], "pair_R": r["pair_recall"], "pair_F1": r["pair_f1"], "retr_thr": r["retr_thr"],
                "retr_F1": r["retr_f1"], "retr_P": r["retr_precision"], "retr_R": r["retr_recall"], "cand_R": r["cand_recall"],
                **{f"AUC_{k}": v for k, v in r["auc_by_negative"].items()}, "note": ""})
print("hybrid: AUC=%.3f pairF1=%.3f retrF1=%.3f" % (r["pair_auc"], r["pair_f1"], r["retr_f1"]))

# %% [markdown]
# ### Experiment 6 — set-overlap similarity (pair-only, no retrieval)
# A non-vector baseline: Jaccard over normalised word sets. Included to show what TF-IDF weighting adds.

# %%
toks = df.title.map(lambda t: set(normalize_title(t).split()))
bt, bv = bench["test"], bench["val"]
jt = np.array([jaccard(toks.iloc[a], toks.iloc[b]) for a, b in zip(bt.i, bt.j)])
jv = np.array([jaccard(toks.iloc[a], toks.iloc[b]) for a, b in zip(bv.i, bv.j)])
thr, _ = best_f1_threshold(bv.y.values, jv); pm = pair_metrics(bt.y.values, jt, thr); auc, ap = auc_ap(bt.y.values, jt)
print(f"Jaccard words: AUC={auc:.3f} AP={ap:.3f} F1={pm['f1']:.3f} thr={thr:.3f}")
jacc_row = dict(Experiment="Exp 6", Representation="word sets", Similarity="Jaccard", AUC=auc, AP=ap, pair_F1=pm["f1"], pair_thr=thr)

# %% [markdown]
# ### Results table

# %%
R = pd.DataFrame(results)
R = pd.concat([R, pd.DataFrame([jacc_row])], ignore_index=True)
R.to_csv(RES / "experiments.csv", index=False)
show = R[["Experiment", "Representation", "Similarity", "dim", "pair_thr", "AUC", "AP", "pair_F1", "retr_thr", "retr_F1", "cand_R", "embed_s"]]
(RES / "experiments.md").write_text(md_table(show))
show

# %%
neg_cols = [c for c in R.columns if c.startswith("AUC_")]
print("AUC split by negative type (test) – how each method handles easy vs hard negatives")
R[["Representation", "Similarity"] + neg_cols].dropna(subset=neg_cols[:1]).round(3)

# %% [markdown]
# **Observations (edit after running):**
# * Compare *Baseline → Exp 1 → Exp 2*: do char n-grams improve recall without hurting precision?
# * Compare lexical vs SBERT in the `AUC_text_hard_neg` column: semantic models tend to be *worse* on
#   near-identical-title negatives (variants look semantically the same).
# * `AUC_random_neg` is near 1 for everything → random negatives say nothing about quality; *hard* negatives do.
# * Exp 4: cosine vs dot vs L2 — with un-normalised SBERT vectors, length differences can add noise.
# * Pair-F1 on the curated benchmark is much higher than retrieval-F1 on the full split: the real task
#   (find the group among thousands of listings) has far more hard negatives than a balanced benchmark.

# %% [markdown]
# ## 4. Threshold analysis
# Using the best retrieval method on **val**: sweep the cosine threshold and track precision, recall, F1
# and the raw counts of false positives / false negatives.

# %%
best_name = R[R.metric == "cos"].sort_values("retr_F1", ascending=False).iloc[0]   # cosine-type scores only
print("best by retrieval-F1 (cosine-based):", best_name.Representation)
Eb = E[best_name.key]
sv = pair_scores(Eb, bv.i.values, bv.j.values); yv = bv.y.values
thrs = np.linspace(np.quantile(sv, .02), np.quantile(sv, .98), 60)
T = threshold_table(yv, sv, thrs)
t_star, f1_star = best_f1_threshold(yv, sv)

fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
ax[0].plot(T.thr, T.precision, label="precision"); ax[0].plot(T.thr, T.recall, label="recall"); ax[0].plot(T.thr, T.f1, label="F1", lw=2)
ax[0].axvline(t_star, ls="--", c="k"); ax[0].set_xlabel("threshold"); ax[0].legend(); ax[0].set_title("Pair metrics vs threshold (val)")
ax[1].plot(T.thr, T.fp, c="tab:red", label="false positives"); ax[1].plot(T.thr, T.fn, c="tab:blue", label="false negatives")
ax[1].axvline(t_star, ls="--", c="k"); ax[1].set_xlabel("threshold"); ax[1].legend(); ax[1].set_title("Error counts vs threshold (val)")
ax[2].plot(T.recall, T.precision); ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision"); ax[2].set_title("Precision-recall curve (val)")
fig.tight_layout(); fig.savefig(RES / "b_threshold_analysis.png", dpi=110); plt.show()

# retrieval-F1 vs threshold
iv, jv_, ssv = topk_pairs(Eb, rows["val"], C.TOPK); yrv = (lab[iv] == lab[jv_]).astype(int)
grid = np.linspace(np.quantile(ssv, .3), ssv.max(), 60)
rf = [retrieval_f1(iv, jv_, ssv, yrv, t, rows["val"], gs, N) for t in grid]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(grid, [r["f1"] for r in rf], label="F1"); ax.plot(grid, [r["precision"] for r in rf], label="precision"); ax.plot(grid, [r["recall"] for r in rf], label="recall")
ax.set_xlabel("threshold"); ax.set_title("Retrieval (Kaggle-style) metrics vs threshold (val)"); ax.legend(); fig.tight_layout()
fig.savefig(RES / "b_retrieval_threshold.png", dpi=110); plt.show()
sel = T.iloc[np.linspace(0, len(T) - 1, 8).astype(int)][["thr", "precision", "recall", "f1", "fp", "fn"]]
sel.to_csv(RES / "threshold_table.csv", index=False); sel.round(3)

# %% [markdown]
# **How the final threshold is chosen:** maximise F1 on **val**, then freeze it and report on **test**
# (never tune on test). Two thresholds are chosen — one for the pair benchmark and one for the
# retrieval metric — because they have different class balance (the retrieval candidate list contains far
# more negatives, so its optimal threshold is higher). A **higher** threshold ⇒ fewer false positives, more false
# negatives (precision↑ recall↓); choose by cost of each error in the application. For a marketplace
# de-duplication system, wrongly merging two products (FP) is usually costlier than missing a duplicate (FN), so a
# precision-leaning threshold could be preferred over the F1-optimal one.

# %% [markdown]
# ## 5. Error analysis (test split, best method)

# %%
st_ = pair_scores(Eb, bt.i.values, bt.j.values); t_pair = best_f1_threshold(bv.y.values, sv)[0]
E_df = bt.copy(); E_df["score"] = st_
E_df["title_i"] = df.title.values[E_df.i]; E_df["title_j"] = df.title.values[E_df.j]
ni, nj = [df.title.map(lambda t: numbers(normalize_title(t))).values[E_df[c]] for c in ("i", "j")]
E_df["numbers_differ"] = [bool(a and b and a != b) for a, b in zip(ni, nj)]
E_df["word_overlap"] = [jaccard(set(normalize_title(a).split()), set(normalize_title(b).split())) for a, b in zip(E_df.title_i, E_df.title_j)]
E_df["pred"] = (E_df.score >= t_pair).astype(int)
tp = E_df[(E_df.y == 1) & (E_df.pred == 1)].sort_values("score", ascending=False)
fp = E_df[(E_df.y == 0) & (E_df.pred == 1)].sort_values("score", ascending=False)
fn = E_df[(E_df.y == 1) & (E_df.pred == 0)].sort_values("score")
print(f"TP={len(tp)}  FP={len(fp)}  FN={len(fn)}  (threshold={t_pair:.3f})")
cols = ["score", "title_i", "title_j", "kind", "numbers_differ", "word_overlap"]
pd.set_option("display.max_colwidth", 70); pd.set_option("display.width", 250)
print("\n=== Correct matches (TP) ==="); print(tp[cols].head(4).to_string(index=False))
print("\n=== Wrong matches (FP) ===");   print(fp[cols].head(8).to_string(index=False))
print("\n=== Missed matches (FN) ===");  print(fn[cols].head(8).to_string(index=False))
pd.concat([tp.head(30).assign(case="TP"), fp.head(50).assign(case="FP"), fn.head(50).assign(case="FN")])[["case"] + cols].to_csv(RES / "error_examples.csv", index=False)

# %%
print("FP by negative type:", fp.kind.value_counts().to_dict())
print(f"FP where numbers differ (sizes/models mismatch): {fp.numbers_differ.mean():.1%}   vs TP: {tp.numbers_differ.mean():.1%}")
print(f"FN with word overlap < 0.2 (disjoint titles):     {(fn.word_overlap < .2).mean():.1%}")
# who fixes whom: lexical vs semantic
if C.SBERT in E or True:
    Es_ = E.get(C.SBERT, get_embedding(df, C.SBERT, tag)); Ec_ = E.get("tfidf_char", get_embedding(df, "tfidf_char", tag))
    s_s, s_c = pair_scores(Es_, bt.i.values, bt.j.values), pair_scores(Ec_, bt.i.values, bt.j.values)
    th_s = best_f1_threshold(bv.y.values, pair_scores(Es_, bv.i.values, bv.j.values))[0]
    th_c = best_f1_threshold(bv.y.values, pair_scores(Ec_, bv.i.values, bv.j.values))[0]
    only_sbert = bt[(bt.y == 1) & (s_s >= th_s) & (s_c < th_c)]; only_char = bt[(bt.y == 1) & (s_c >= th_c) & (s_s < th_s)]
    print(f"\ntrue matches found ONLY by SBERT: {len(only_sbert)} | ONLY by char-TF-IDF: {len(only_char)}  -> complementary errors justify the hybrid")
    for nm, d in [("only SBERT", only_sbert), ("only char TF-IDF", only_char)]:
        print(f"\n--- {nm} ---")
        for r in d.head(3).itertuples(): print(f"  A: {df.title.iloc[r.i][:70]}\n  B: {df.title.iloc[r.j][:70]}\n")

# %% [markdown]
# **Worked analysis of the examples (edit with your own examples from the printout above):**
#
# *Example format from the task description:* `"Samsung Galaxy Buds Pro Original"` vs `"Samsung Wireless Earbuds Pro"`
# → a lexical model gives a low score (only "samsung", "pro" overlap) = **false negative**, an SBERT model maps
# "earbuds ≈ buds", "wireless" → likely **correct match**. Conversely, `"Case iPhone 11 Pro Max"` vs
# `"Case iPhone 11 Pro"` share almost everything → **false positive** for every text model: the discriminative
# detail ("Max") is a single token.
#
# | Error type | Typical cause | Possible fix |
# |---|---|---|
# | FP: variants (size/colour/model) | title differs only in 1 token; embeddings are smooth | number/variant features, learned pair classifier, image colour |
# | FP: generic titles ("tas wanita murah") | high similarity on common words | IDF weighting, length features, add image |
# | FN: disjoint titles | seller uses different words/language | semantic embeddings, translation, image |
# | FN: very short titles | too little signal | rely on the image modality |
# | FN: title contains extra keyword spam | spam dilutes the cosine | token-level matching / BM25-style scoring |

# %% [markdown]
# ## 6. Conclusions
# * Fill in the final table values; state which representation wins on **retrieval-F1** and **hard negatives**, not only AUC.
# * Lexical (char n-gram) and semantic (SBERT) signals are complementary → hybrid ≥ both (verify in table).
# * Text alone has a ceiling: variants and disjoint titles cannot be solved from titles → motivates Part C and the Finale.
# * **Answers to the guiding questions** — *Mathematical representation:* sparse TF-IDF vectors (weighted token counts) or dense
#   semantic vectors; similarity = angle between them. *Why TF-IDF works:* rare tokens (brands/models) get high weight
#   and cosine ignores length; *why it fails:* no semantics, brittle to spelling/language. *Char n-grams:* yes, robust to morphology/typos.
#   *Why embeddings can win:* paraphrase/multilingual generalisation. *Near-identical titles, different products:* false positives.
#   *Same product, different titles:* false negatives.

# %%
json.dump({"best_by_retrieval_f1": best_name.Representation, "pair_threshold": float(t_pair)}, open(RES / "summary.json", "w"), indent=2)
print("saved ->", RES)
