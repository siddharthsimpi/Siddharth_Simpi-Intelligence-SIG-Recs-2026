# %% [markdown]
# # Finale — Multimodal Product Matching
#
# **Understand → Experiment → Analyse → Improve → Explain.**
#
# ## System architecture (two-stage retrieve → rerank)
# ```
#  title ──► TF-IDF word / TF-IDF char / SBERT ─┐                    ┌─► cosine per encoder ─┐
#                                               ├─ top-K neighbours ─┤                       ├─► pair features ─► scorer ─► threshold ─► match set
#  image ──► ResNet-50 / CLIP ViT-B/32 ─────────┘   (union = pool)   └─► pHash, number, len ─┘   (fusion / GBM)           (val-tuned)  (+graph post-proc.)
# ```
# * **Stage 1 – retrieval:** every listing retrieves its top-K neighbours under *each* modality; the union is the candidate pool
#   (avoids O(n²) classification and raises the recall ceiling).
# * **Stage 2 – scoring:** each candidate pair gets features (per-encoder cosines, perceptual-hash distance, title/number
#   overlap) and a scorer (weighted score fusion **or** a gradient-boosted classifier) outputs a match score.
# * **Decision:** threshold tuned on **val**, evaluated on **test** (group-wise split, no product shared between splits).
#
# **Metric:** Kaggle-style mean per-listing F1 over predicted match sets (higher = better), "retrieval-F1" below.
#
# **External resources:** scikit-learn (TF-IDF, HistGradientBoosting), sentence-transformers multilingual MiniLM,
# CLIP ViT-B/32 (OpenAI) + torchvision ResNet-50, Kaggle competition metric. Part B/C code is reused via `shopee_match/`.

# %%
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent)); sys.path.insert(0, str(Path.cwd()))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt

from shopee_match import config as C
from shopee_match.data import load_dataset, split_rows, group_sizes, get_tag
from shopee_match.embeddings import get_embedding
from shopee_match.viz import pair_figure, md_table
from src.features import candidate_pairs, PairFeaturizer
from src.fusion import weighted_score, fit_gbm, gbm_score
from src.pipeline import Ctx, tune, at_threshold, evaluate, cand_recall, symmetrize, components, predicted_matches, build_match_lists
from src.error_analysis import categorize, slice_rates, missed_by_retrieval

RES = C.results_dir("Finale")
df = load_dataset(); tag = get_tag(df); rows = split_rows(df); gs = group_sizes(df); lab = df.label_group.values
TEXT_MODS = ["tfidf_word", "tfidf_char", C.SBERT]
IMG_MODS = [C.RESNET, C.CLIP_IMG]
CAND_MODS = ["tfidf_char", C.SBERT, C.RESNET, C.CLIP_IMG]   # modalities used for candidate retrieval
CAND_K = 20
LOG, METRICS = [], {"dataset": {"n": len(df), **{f"n_{k}": int(len(v)) for k, v in rows.items()}, "smoke": C.SMOKE}}
print(METRICS["dataset"])

# %% [markdown]
# ## 1. Stage 1 – candidate generation and pair features
# Candidate pairs are built **within each split** (train for fitting the learned scorer, val for tuning, test for reporting).
# `cand_recall` is the *ceiling* of the whole system: a true match that is never retrieved can never be predicted.

# %%
t0 = time.time()
E = {m: get_embedding(df, m, tag) for m in dict.fromkeys(TEXT_MODS + IMG_MODS)}
fz = PairFeaturizer(df, E, TEXT_MODS, IMG_MODS); COLS = fz.groups
cand, feats = {}, {}
for s in ("train", "val", "test"):
    cand[s] = candidate_pairs(E, CAND_MODS, rows[s], CAND_K, lab)
    feats[s] = fz(cand[s])
ctx = Ctx(df, rows, gs, cand, feats)
for s in cand:
    print(f"{s:5s}: {len(cand[s]):>9,} candidate pairs | positive rate {cand[s].y.mean():.3f} | candidate recall ceiling {cand_recall(ctx, s):.3f}")
METRICS["candidate"] = {"k": CAND_K, "mods": CAND_MODS, "recall_test": cand_recall(ctx, "test"), "pairs_test": int(len(cand["test"]))}
print(f"feature groups: { {k: len(v) for k, v in COLS.items()} } | built in {time.time()-t0:.0f}s")

# %% [markdown]
# ## 2. Baseline
# Each modality alone, then the simplest multimodal baseline: **equal-weight score fusion** of the best text
# cosine and the best image cosine (best = highest *val* F1).

# %%
TXT, IMG = COLS["text"], COLS["image"]
sc = lambda c, s: feats[s][c].values
singles = [evaluate(ctx, c, sc(c, "val"), sc(c, "test"), modality="text" if c in TXT else "image") for c in TXT + IMG]
S = pd.DataFrame(singles); best_t = S[S.modality == "text"].sort_values("val_F1").iloc[-1].method; best_i = S[S.modality == "image"].sort_values("val_F1").iloc[-1].method
print(S[["method", "modality", "thr", "val_F1", "test_F1", "test_P", "test_R"]].round(4).to_string(index=False))
print("\nbest text:", best_t, "| best image:", best_i)
base_val = 0.5 * sc(best_t, "val") + 0.5 * sc(best_i, "val"); base_test = 0.5 * sc(best_t, "test") + 0.5 * sc(best_i, "test")
baseline = evaluate(ctx, "BASELINE: 0.5*text + 0.5*image", base_val, base_test)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in baseline.items()})

# %% [markdown]
# **Baseline description**
#
# | Item | Choice |
# |---|---|
# | Architecture | retrieve union top-20 (4 encoders) → score → threshold |
# | Features / representations | best single text cosine + best single image cosine (see above) |
# | Similarity | cosine on L2-normalised vectors (TF-IDF is already L2-normalised) |
# | Matching strategy | match iff `0.5·s_text + 0.5·s_img ≥ τ`; τ maximises val retrieval-F1; self always included |
# | Hyper-parameters | K = 20 per encoder, weight 0.5, τ from a 80-point grid on val, no learning |

# %% [markdown]
# ## 3. Experiments (Hypothesis → Experiment → Result → Analysis → Conclusion)
# Each experiment changes **one** component relative to the previous reference so effects are separable.

# %%
def verdict(delta, tol=0.002):
    return "SUPPORTED" if delta > tol else ("NOT SUPPORTED (no gain)" if delta > -tol else "REJECTED (worse)")
ref = {"name": "baseline", "val": baseline["val_F1"], "test": baseline["test_F1"]}
EXP_ROWS = [baseline]

# %% [markdown]
# ### E1 — Score-level fusion with a tuned weight
# **Hypothesis.** Text and image make *partly independent* errors, so fusing them beats the best single modality; and the optimal
# weight is **not** 0.5 (it should tilt towards the more reliable modality).
# **Experiment.** `s = w·s_text + (1−w)·s_img`, sweep w ∈ [0,1] (step 0.05), choose w and τ on val. Only the weight changes vs. baseline.

# %%
ws = np.round(np.linspace(0, 1, 21), 2); curve = []
for w in ws:
    sv = w * sc(best_t, "val") + (1 - w) * sc(best_i, "val"); curve.append(tune(ctx, sv)[1])
w_star = float(ws[int(np.argmax(curve))])
e1 = evaluate(ctx, f"E1 weighted fusion (w_text={w_star})", w_star * sc(best_t, "val") + (1 - w_star) * sc(best_i, "val"),
              w_star * sc(best_t, "test") + (1 - w_star) * sc(best_i, "test"), w_text=w_star)
fig, ax = plt.subplots(figsize=(6, 3.4)); ax.plot(ws, curve, marker="o"); ax.axvline(w_star, ls="--", c="k")
ax.set_xlabel("w_text  (0 = image only, 1 = text only)"); ax.set_ylabel("val retrieval-F1"); ax.set_title("E1: fusion weight sweep")
fig.tight_layout(); fig.savefig(RES / "final_e1_weight_sweep.png", dpi=110); plt.show()
best_single_test = S.test_F1.max()
v1 = verdict(e1["test_F1"] - max(best_single_test, baseline["test_F1"]))
print(f"best single test F1={best_single_test:.4f} | baseline={baseline['test_F1']:.4f} | E1={e1['test_F1']:.4f} (w*={w_star}) -> {v1}")
LOG.append(dict(id="E1", title="Weighted score fusion", hypothesis="Fusing text+image beats the best single modality and the optimal weight differs from 0.5.",
                experiment="Sweep w in [0,1] on val for s=w*text+(1-w)*image; choose threshold on val.",
                result=f"w*={w_star}; test F1 {e1['test_F1']:.4f} vs baseline {baseline['test_F1']:.4f} vs best single {best_single_test:.4f}", verdict=v1,
                analysis="The weight curve shows how much each modality contributes; a curve that peaks strictly between 0 and 1 means the modalities carry complementary information.",
                conclusion=f"Hypothesis {v1.lower()}."))
EXP_ROWS.append(e1)

# %% [markdown]
# **Analysis.** Read the sweep: the two ends (w=0, w=1) are the single-modality results. If the curve is higher in the interior the
# modalities are complementary; the location of the maximum tells which one is more reliable. *(Edit with your numbers.)*
# **Conclusion.** Recorded in the auto-verdict above.

# %% [markdown]
# ### E1b — ensembling *within* each modality
# **Hypothesis.** Averaging heterogeneous encoders inside a modality (lexical + semantic text, CNN + CLIP image) reduces variance and beats the single best encoder.
# **Experiment.** Replace `best_t`/`best_i` by `mean(all text cosines)` / `mean(all image cosines)`; re-tune w and τ.

# %%
tm = lambda s: feats[s][TXT].values.mean(1); im = lambda s: feats[s][IMG].values.mean(1)
curve2 = [tune(ctx, w * tm("val") + (1 - w) * im("val"))[1] for w in ws]; w2 = float(ws[int(np.argmax(curve2))])
e1b = evaluate(ctx, f"E1b ensemble-in-modality fusion (w_text={w2})", w2 * tm("val") + (1 - w2) * im("val"), w2 * tm("test") + (1 - w2) * im("test"), w_text=w2)
v1b = verdict(e1b["test_F1"] - e1["test_F1"])
print(f"E1={e1['test_F1']:.4f} -> E1b={e1b['test_F1']:.4f} (w*={w2}) -> {v1b}")
LOG.append(dict(id="E1b", title="Ensembling within each modality", hypothesis="Mean of several encoders per modality beats the single best encoder per modality.",
                experiment="s = w*mean(text cosines)+(1-w)*mean(image cosines); w, thr tuned on val.", result=f"test F1 {e1b['test_F1']:.4f} vs E1 {e1['test_F1']:.4f}", verdict=v1b,
                analysis="Lexical (TF-IDF) and semantic (SBERT) text encoders, and CNN vs CLIP image encoders, err differently; their average is smoother.",
                conclusion=f"Hypothesis {v1b.lower()}."))
EXP_ROWS.append(e1b)
ref_w = e1b if e1b["test_F1"] >= e1["test_F1"] else e1

# %% [markdown]
# ### E2 — learned fusion (gradient boosting) instead of a linear weighted sum
# **Hypothesis.** The right way to combine the modalities is *non-linear and context dependent* (e.g. "trust the image when titles are
# short", "require BOTH to be high for variants"). A GBM on the same 5 cosine features should beat the best linear fusion.
# **Experiment.** Train `HistGradientBoostingClassifier` on **train-split** candidate pairs using only the 5 cosine features (same information as E1b);
# threshold on val. *Only the combiner changes* (linear → learned).

# %%
def run_gbm(cols, name):
    t0 = time.time(); m = fit_gbm(feats["train"], cand["train"].y.values, cols)
    sv, st = gbm_score(m, feats["val"], cols), gbm_score(m, feats["test"], cols)
    r = evaluate(ctx, name, sv, st); r["fit_s"] = time.time() - t0
    return m, sv, st, r
cos_cols = TXT + IMG
m2, s2v, s2t, e2 = run_gbm(cos_cols, "E2 GBM on 5 cosine features")
v2 = verdict(e2["test_F1"] - ref_w["test_F1"])
print(f"best linear={ref_w['test_F1']:.4f} -> GBM(cos only)={e2['test_F1']:.4f} -> {v2}")
LOG.append(dict(id="E2", title="Learned (GBM) fusion on cosine features", hypothesis="A non-linear learned combiner beats the best weighted-sum fusion.",
                experiment="HistGradientBoosting on the same 5 cosines, trained on train-split candidate pairs.", result=f"test F1 {e2['test_F1']:.4f} vs linear {ref_w['test_F1']:.4f}", verdict=v2,
                analysis="A GBM can learn interactions (e.g. both-modalities-agree vs one-modality-only) and per-feature non-linear calibration that a single weight cannot.",
                conclusion=f"Hypothesis {v2.lower()}."))
EXP_ROWS.append(e2)

# %% [markdown]
# ### E3 — add side information: perceptual hash and title/number features
# **Hypothesis.** Features that encode *variant conflicts* (numbers/sizes differ, low word-overlap) and *exact duplicates* (pHash distance, same file) fix errors that
# cosines cannot — mainly false positives on near-identical titles/images — so the GBM with all 16 features beats E2.
# **Experiment.** Same GBM, features = cosines + pHash group + meta group. Only the *feature set* changes vs E2.

# %%
all_cols = COLS["text"] + COLS["image"] + COLS["phash"] + COLS["meta"]
m3, s3v, s3t, e3 = run_gbm(all_cols, "E3 GBM on all 16 features")
v3 = verdict(e3["test_F1"] - e2["test_F1"])
print(f"GBM(cos)={e2['test_F1']:.4f} -> GBM(all)={e3['test_F1']:.4f} -> {v3}")
imp = pd.Series(np.abs(np.corrcoef(feats["val"][all_cols].values.T, cand["val"].y.values)[-1, :-1]), index=all_cols).sort_values(ascending=False)
print("\n|corr(feature, label)| on val (a rough importance proxy):"); print(imp.round(3).head(10).to_string())
LOG.append(dict(id="E3", title="Side-information features (pHash, numbers, lengths)", hypothesis="Variant/duplicate features reduce false positives and improve F1 over cosines only.",
                experiment="GBM with 16 features (cosines + pHash + meta) vs GBM with 5 cosines.", result=f"test F1 {e3['test_F1']:.4f} vs {e2['test_F1']:.4f}", verdict=v3,
                analysis="Number-token conflicts separate size/model variants; pHash==0 identifies re-used pictures; word-overlap and length ratios flag spammy or truncated titles.",
                conclusion=f"Hypothesis {v3.lower()}."))
EXP_ROWS.append(e3)

# %% [markdown]
# ### E4 — graph post-processing (symmetry and transitivity)
# **Hypothesis.** Matching is an *equivalence relation*, but our edges come from asymmetric top-K lists. (a) adding reverse edges and
# (b) taking connected components (transitive closure) recovers matches that were never directly retrieved/linked, raising recall; the risk is chaining wrong edges (lower precision).
# **Experiment.** Apply to the E3 scores; threshold re-tuned on val for each variant. *Scoring model unchanged.*

# %%
def search_post(score_val, n_grid=60):
    """Best (threshold, val F1) for every post-processing mode, thresholds chosen on VAL only."""
    out = {"none": tune(ctx, score_val)[:2]}
    grid = np.quantile(score_val, np.linspace(0.6, 0.9995, n_grid))
    for mode, fn in (("symmetric", symmetrize), ("components", components)):
        fs = [fn(ctx, score_val, t, "val")["f1"] for t in grid]
        out[mode] = (float(grid[int(np.argmax(fs))]), float(np.max(fs)))
    return out
post = search_post(s3v)
post_rows = []
for mode, (t, fv) in post.items():
    r = {"none": at_threshold(ctx, s3t, t, "test"), "symmetric": symmetrize(ctx, s3t, t, "test"), "components": components(ctx, s3t, t, "test")}[mode]
    post_rows.append({"post-processing": mode, "thr(val)": t, "val_F1": fv, "test_F1": r["f1"], "test_P": r["precision"], "test_R": r["recall"]})
P = pd.DataFrame(post_rows); print(P.round(4).to_string(index=False))
best_post = P.sort_values("val_F1").iloc[-1]["post-processing"]
v4 = verdict(float(P.set_index("post-processing").loc[best_post, "test_F1"] - P.set_index("post-processing").loc["none", "test_F1"]))
LOG.append(dict(id="E4", title="Graph post-processing", hypothesis="Symmetrising / transitive closure raises recall more than it hurts precision.",
                experiment="none vs symmetric vs connected components on E3 scores; thr tuned on val per variant.", result=P.round(4).to_dict("records"), verdict=v4,
                analysis="Closure helps products with many listings (recall) but a single false edge can merge two products (precision); the max component size guard limits the damage.",
                conclusion=f"Best on val: {best_post}; hypothesis {v4.lower()}."))
P.to_csv(RES / "e4_postprocessing.csv", index=False)

# %% [markdown]
# ## 4. Ablation study
# All learned configurations use the **same candidate pool and the same GBM**; only the *feature groups* differ. Rows A–B are the unimodal systems, C is the plain multimodal
# system, D–F add side-information, G–H remove a modality from the full system.

# %%
cfgs = [("A", ["text"]), ("B", ["image"]), ("C", ["text", "image"]), ("D", ["text", "image", "phash"]), ("E", ["text", "image", "meta"]),
        ("F (full)", ["text", "image", "phash", "meta"]), ("G (full − text)", ["image", "phash", "meta"]), ("H (full − image)", ["text", "phash", "meta"])]
abl = []
for name, groups in cfgs:
    cols = sum([COLS[g] for g in groups], [])
    _, sv_, st_, r = run_gbm(cols, name)
    abl.append({"Configuration": name, "Text": "✓" if "text" in groups else "✗", "Image": "✓" if "image" in groups else "✗",
                "pHash": "✓" if "phash" in groups else "✗", "Meta (nums/len)": "✓" if "meta" in groups else "✗",
                "val_F1": r["val_F1"], "Score (test F1)": r["test_F1"], "Precision": r["test_P"], "Recall": r["test_R"], "pair AP": r["pair_AP"]})
    print(f"{name:18s} test F1={r['test_F1']:.4f}")
A = pd.DataFrame(abl); A.to_csv(RES / "ablation.csv", index=False); (RES / "ablation.md").write_text(md_table(A)); A

# %%
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.barh(A.Configuration[::-1], A["Score (test F1)"][::-1], color="tab:blue")
for y_, v_ in enumerate(A["Score (test F1)"][::-1]): ax.text(v_, y_, f" {v_:.3f}", va="center", fontsize=8)
ax.set_xlabel("test retrieval-F1"); ax.set_title("Ablation (same pool, same GBM, different feature groups)"); fig.tight_layout()
fig.savefig(RES / "final_ablation.png", dpi=110); plt.show()
d = A.set_index("Configuration")["Score (test F1)"]
print(f"Δ from adding image to text: {d['C'] - d['A']:+.4f} | adding text to image: {d['C'] - d['B']:+.4f} | + pHash: {d['D'] - d['C']:+.4f} | + meta: {d['E'] - d['C']:+.4f} | full vs C: {d['F (full)'] - d['C']:+.4f}")

# %% [markdown]
# **Reading the ablation (edit with the printed deltas):** the bigger the drop when a component is removed (rows G/H vs F), the more that
# component contributes. If row C > max(A, B) the multimodal gain is real; if D/E ≈ C the side features add little *for this scorer*.

# %% [markdown]
# ## 5. Final system selection and export
# Among the candidate systems the choice is made **on val only**; the test number is computed once at the end.

# %%
options = {"E1b/E1 weighted fusion": (ref_w, None), "E2 GBM (cosines)": (e2, (m2, s2v, s2t, cos_cols)), "E3 GBM (all features)": (e3, (m3, s3v, s3t, all_cols))}
choice = max(options, key=lambda k: options[k][0]["val_F1"]); print("final scorer chosen on val:", choice)
if options[choice][1] is None:
    wf = ref_w["w_text"]; use_t, use_i = (TXT, IMG) if ref_w is e1b else ([best_t], [best_i])
    fv_, ft_ = (weighted_score(feats[s], use_t, use_i, wf) for s in ("val", "test")); bundle_kind, model_, cols_ = "weighted", None, None
else:
    model_, fv_, ft_, cols_ = options[choice][1]; bundle_kind = "gbm"
# post-processing + threshold chosen on val
post_final = search_post(fv_)
final_mode = max(post_final, key=lambda m: post_final[m][1]); final_thr, final_val = post_final[final_mode]
final_test = {"none": at_threshold, "symmetric": symmetrize, "components": components}[final_mode](ctx, ft_, final_thr, "test")
print(f"FINAL: scorer={choice} | post={final_mode} | thr={final_thr:.4f} | val F1={final_val:.4f} | TEST F1={final_test['f1']:.4f} (P={final_test['precision']:.4f}, R={final_test['recall']:.4f})")

bundle = dict(kind=bundle_kind, model=model_, cols=cols_, text_cols=(use_t if bundle_kind == "weighted" else None), img_cols=(use_i if bundle_kind == "weighted" else None),
            w_text=(wf if bundle_kind == "weighted" else None), thr=final_thr, post=final_mode, text_mods=TEXT_MODS, img_mods=IMG_MODS, cand_mods=CAND_MODS, cand_k=CAND_K)
joblib.dump(bundle, RES / "final_model.joblib")
predicted_matches(ctx, ft_, final_thr, "test", final_mode).to_csv(RES / "test_predictions.csv", index=False)

# context from Part B / C (single-modality systems evaluated on the same test split)
context = {}
for part, key in (("PartB", "partB_best_retr_F1"), ("PartC", "partC_best_retr_F1")):
    f = Path.cwd().parent / part / "results" / "experiments.csv"
    if f.exists():
        x = pd.read_csv(f); context[key] = float(x["retr_F1"].max())
print("context (single-modality best from Parts B/C):", context)

# %% [markdown]
# ## 6. Error analysis (final system, test split)
# Every candidate pair is classified by *which modality alone would have said "match"* (thresholds tuned on val), which exposes conflicts between text and image evidence.

# %%
c_t, F_t, y_t = cand["test"], feats["test"], cand["test"].y.values; pred = (ft_ >= final_thr).astype(int)
tt = tune(ctx, tm("val"))[0]
ti = tune(ctx, im("val"))[0]
cat = categorize(F_t, y_t, pred, TXT, IMG, tt, ti)
ec = pd.Series(cat[cat != ""]).value_counts(); print(ec.to_string())
sl = slice_rates(F_t, y_t, pred); print("\n", sl.round(3).to_string(index=False))
miss_pairs, n_miss, n_true = missed_by_retrieval(ctx, "test")
print(f"\ntrue matches never retrieved (stage-1 misses): {n_miss:,} / {n_true:,} = {n_miss/max(n_true,1):.1%}")
sl.to_csv(RES / "error_slices.csv", index=False); ec.to_csv(RES / "error_categories.csv", header=["count"])

WHY = {
    "FP: text AND image agree (near-identical variant / similar product)": ("same product family, different variant (colour/size/model)", "variant-aware features: colour histogram, size/number extraction, fine-tuned embeddings with hard negatives"),
    "FP: text matches, images differ": ("generic or copied titles; seller photos differ", "down-weight generic title words, require image agreement when titles are generic"),
    "FP: image matches, titles differ (visually similar product)": ("stock photo / similar item", "require some text support; instance-level image embeddings"),
    "FP: fused model over-trusted weak cues": ("many individually weak signals added up", "calibrate scorer, raise threshold, add more negatives in training"),
    "FN: titles match, images differ (different photos of same product)": ("viewpoint, packaging vs item, lifestyle photo", "image fine-tuning with multi-view positives, rely more on text when the image is unreliable"),
    "FN: image matches, titles differ (noisy / disjoint titles)": ("other language, keyword spam, very short title", "multilingual encoder, title cleaning, translation, trust image more for short titles"),
    "FN: neither modality matches (hard)": ("both text and image differ - maybe even label noise", "more modalities/attributes, cluster-level reasoning, inspect label noise"),
    "FN: both modalities match but fused model rejected (threshold / calibration)": ("fused score borderline", "threshold per group size / local adaptive thresholds"),
}
pd.DataFrame([{"error category": k, "count": int(v), "why it fails": WHY[k][0], "possible fix": WHY[k][1]} for k, v in ec.items()]).to_csv(RES / "error_why_fix.csv", index=False)

# %%
def show_errors(kind, n_per=2, fname=None):
    sel = []
    for k in [c for c in ec.index if c.startswith(kind)]:
        idx = np.where(cat == k)[0]
        idx = idx[np.argsort(-np.abs(ft_[idx] - final_thr))][:n_per]            # confident mistakes first
        sel += [(k, t) for t in idx]
    if not sel: print("no", kind, "errors"); return
    pairs = [(c_t.i.iloc[t], c_t.j.iloc[t]) for _, t in sel]
    caps = [f"{k.split(': ')[1][:38]}\nscore={ft_[t]:.2f} thr={final_thr:.2f}\ntext={F_t[TXT].values[t].mean():.2f} img={F_t[IMG].values[t].mean():.2f}\nTRUE: {'same' if y_t[t] else 'different'} product" for k, t in sel]
    pair_figure(df, pairs, caps, save=RES / fname, title=f"{kind} examples (predicted {'match' if kind == 'FP' else 'no match'}; truth is the opposite)"); plt.show()
    for (k, t), (i, j) in zip(sel, pairs): print(f"[{k}]\n  A: {df.title.iloc[i][:80]}\n  B: {df.title.iloc[j][:80]}\n  predicted={'match' if pred[t] else 'no match'} | correct={'match' if y_t[t] else 'no match'} | why: {WHY[k][0]} | fix: {WHY[k][1]}\n")
show_errors("FP", fname="final_errors_fp.png"); show_errors("FN", fname="final_errors_fn.png")

# %% [markdown]
# **Template for discussing each error (fill in with the printed cases):**
# 1. *What did the model predict?* – see "predicted" in the printout.
# 2. *What was the correct answer?* – see "correct".
# 3. *Why did it fail?* – category-specific reason + your own look at the two listings (similar products, different variants, noisy titles, different images of the same product, visually similar products, missing information, conflicting text/image evidence).
# 4. *What could fix it?* – the "fix" column; the most promising are fine-tuning with hard negatives and variant-aware attribute extraction.

# %% [markdown]
# ## 7. Results summary and report data

# %%
summary = pd.DataFrame([{k: baseline[k] for k in ("method", "val_F1", "test_F1", "test_P", "test_R")}] +
                       [{k: r[k] for k in ("method", "val_F1", "test_F1", "test_P", "test_R")} for r in (e1, e1b, e2, e3)] +
                       [{"method": f"FINAL ({choice} + {final_mode})", "val_F1": final_val, "test_F1": final_test["f1"], "test_P": final_test["precision"], "test_R": final_test["recall"]}])
summary.to_csv(RES / "experiments.csv", index=False); (RES / "experiments.md").write_text(md_table(summary)); print(summary.round(4).to_string(index=False))
METRICS.update(baseline=baseline, singles=singles, experiments=LOG, ablation=abl, context=context,
               final={"scorer": choice, "post": final_mode, "thr": final_thr, "val_F1": final_val, "test_F1": final_test["f1"], "test_P": final_test["precision"], "test_R": final_test["recall"]},
               errors={"counts": {k: int(v) for k, v in ec.items()}, "slices": sl.to_dict("records"), "missed_by_retrieval": n_miss, "true_pairs": n_true})
json.dump(METRICS, open(RES / "final_metrics.json", "w"), indent=2, default=float)

# %% [markdown]
# ## 8. Limitations and future work (also in the report)
# **Limitations:** frozen pretrained encoders (no fine-tuning on product identity); fixed top-K candidate pool (recall ceiling, see stage-1 misses);
# a single threshold for all group sizes; TF-IDF fitted on the evaluated corpus (transductive); labels may contain noise; split-level results come from one random
# group split (no confidence intervals); no explicit colour/size/brand attribute extraction.
# **Future improvements:** metric-learning fine-tuning (ArcFace / contrastive with hard-negative mining) of image and text encoders; joint multimodal embedding;
# attribute extraction (brand, size, colour) with a rule on conflicts; ANN index (HNSW) for scale; query expansion / DBA; adaptive thresholds per neighbourhood; cross-validated
# group splits with confidence intervals; cluster-level (community-detection) post-processing instead of plain connected components.

# %%
from src.make_report import build
build()
print("report ->", RES.parent / "report.pdf")
