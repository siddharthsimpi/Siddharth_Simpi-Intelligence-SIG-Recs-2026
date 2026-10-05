# %% [markdown]
# # Part C — Image-Based Product Matching
#
# **Pipeline:** `image → preprocessing → pretrained CNN/ViT → embedding → similarity → threshold → match`
#
# No vision model is trained: we use **pretrained encoders** as fixed feature extractors and study
# *what the embeddings capture, how to compare them, where they fail and what it costs*.
#
# Protocol: group-wise train/val/test split; thresholds chosen on **val**, all reported numbers on **test**;
# identical pair benchmark as Part B (so Part B and Part C numbers are directly comparable).
#
# **External resources:** torchvision pretrained ResNet-50 / EfficientNet-B0 (ImageNet), OpenAI CLIP ViT-B/32
# via HuggingFace `transformers` (Radford et al., 2021), perceptual hash column from the dataset.

# %%
import sys, json, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from shopee_match import config as C
from shopee_match.data import load_dataset, split_rows, group_sizes, get_tag
from shopee_match.embeddings import get_embedding, embedding_meta, load_rgb, l2norm
from shopee_match.retrieval import pair_scores, topk_pairs, topk_search
from shopee_match.pairs import get_benchmark, phash_u64, pair_hamming
from shopee_match.evaluate import evaluate_embedding
from shopee_match.metrics import (auc_ap, best_f1_threshold, pair_metrics, threshold_table, retrieval_f1)
from shopee_match.viz import image_grid, pair_figure, md_table

RES = C.results_dir("PartC")
df = load_dataset(); tag = get_tag(df); rows = split_rows(df); gs = group_sizes(df)
lab, N = df.label_group.values, len(df)
bench = get_benchmark(df, rows, tag); bv, bt = bench["val"], bench["test"]
print({k: len(v) for k, v in rows.items()})

# %% [markdown]
# ## 1. Image preprocessing
# *Design decisions:* (a) **resize the whole image to 224×224 without centre-cropping** (product photos have the
# object anywhere; cropping can cut it off) – a small aspect distortion is accepted; (b) normalise with the
# statistics the backbone was trained with (ImageNet for CNNs, CLIP's own for CLIP); (c) unreadable/missing
# files become a neutral grey image instead of crashing, so every listing still gets an embedding.

# %%
import os
missing = int((~pd.Series(df.path).map(os.path.exists)).sum())
print(f"missing image files: {missing}")
ids = df.sample(4, random_state=1).index.tolist()
fig, ax = plt.subplots(2, 4, figsize=(12, 6))
for k, i in enumerate(ids):
    im = load_rgb(df.path.iloc[i]); ax[0, k].imshow(im); ax[0, k].set_title(f"original {im.size}", fontsize=8); ax[0, k].axis("off")
    ax[1, k].imshow(im.resize((224, 224))); ax[1, k].set_title("model input 224x224", fontsize=8); ax[1, k].axis("off")
fig.tight_layout(); fig.savefig(RES / "c_preprocessing.png", dpi=110); plt.show()

# %% [markdown]
# ## 2. What is an image embedding? (concept + demonstration)
# A pretrained network is a function `f: image → ℝᵈ`. The last layers were trained to make *semantically
# similar images produce nearby vectors* (so a classifier on top can separate classes). Removing the classifier
# head leaves a **d-dimensional descriptor** (ResNet-50: 2048 after global-average-pooling; CLIP ViT-B/32: 512).
# Because it is just a vector, similarity search = geometry: after L2-normalisation, **cosine similarity = dot product**
# measures the angle between descriptors, and a **threshold** on that score converts similarity into a match
# decision. Two images of the same product can differ in embedding (viewpoint/background/crop change *what the
# network encodes*) and different products can be close (the network encodes *category/appearance*, not
# *instance identity*, because it was never trained to distinguish individual sellers' SKUs).

# %%
E_demo = get_embedding(df, C.RESNET, tag)
sc = pair_scores(E_demo, bt.i.values, bt.j.values)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
for k, c in [("positive", "tab:green"), ("random_neg", "tab:gray"), ("image_hard_neg", "tab:red"), ("text_hard_neg", "tab:orange")]:
    ax.hist(sc[bt.kind.values == k], bins=40, alpha=.5, density=True, label=k, color=c)
ax.set_xlabel("cosine similarity of image embeddings"); ax.legend(); ax.set_title(f"{C.RESNET}: similarity distributions (test pairs)")
fig.tight_layout(); fig.savefig(RES / "c_similarity_distribution.png", dpi=110); plt.show()
print("norms of raw embedding:", np.linalg.norm(get_embedding(df, C.RESNET, tag, normalize=False)[:5], axis=1).round(2))

# %% [markdown]
# ## 3. Baselines and experiments

# %%
results, E = [], {}
def run(exp, name, model, sim="cosine", metric="cos", norm=True, emb=None, note=""):
    key = emb[0] if emb else name
    Ei = emb[1] if emb else get_embedding(df, name, tag, normalize=norm)
    E[key] = Ei
    t0 = time.perf_counter(); topk_pairs(Ei, rows["test"], C.TOPK, metric); search_s = time.perf_counter() - t0
    r = evaluate_embedding(Ei, df, rows, bench, gs, metric=metric)
    meta = embedding_meta(tag, name)
    results.append({"Experiment": exp, "Model": model, "key": key, "metric": metric, "dim": r["dim"], "Similarity": sim, "pair_thr": r["pair_thr"],
                    "AUC": r["pair_auc"], "AP": r["pair_ap"], "pair_F1": r["pair_f1"], "retr_thr": r["retr_thr"],
                    "retr_F1": r["retr_f1"], "retr_P": r["retr_precision"], "retr_R": r["retr_recall"], "cand_R": r["cand_recall"],
                    "embed_s": meta.get("seconds", np.nan), "emb_MB": N * r["dim"] * 4 / 1e6, "search_s(test)": search_s,
                    **{f"AUC_{k}": v for k, v in r["auc_by_negative"].items()}, "note": note})
    print(f"{exp:10s} {model:34s} dim={r['dim']:5d} AUC={r['pair_auc']:.3f} pairF1={r['pair_f1']:.3f} retrF1={r['retr_f1']:.3f}  search={search_s:.1f}s")

# %% [markdown]
# ### Baseline 0 — perceptual hash (no deep learning)
# The dataset ships a 64-bit pHash. Treated as a ±1 vector, cosine = 1 − 2·Hamming/64. It detects
# (near-)duplicate pictures only; it is the floor any learned embedding must beat.

# %%
run("Baseline 0", "phash_bits", "pHash (64 bit)", "Hamming≈cosine")

# %% [markdown]
# ### Baseline — ResNet-50 (ImageNet), global-average-pooled, cosine
# *Why:* the standard, well-understood feature extractor; strong generic visual features.

# %%
run("Baseline", C.RESNET, "ResNet-50 GAP")

# %% [markdown]
# ### Experiment 1 — different backbone: EfficientNet-B0
# *Hypothesis:* a more parameter-efficient CNN (1280-d) gives comparable quality at lower cost.

# %%
run("Exp 1", C.EFFNET, "EfficientNet-B0")

# %% [markdown]
# ### Experiment 2 — CLIP ViT-B/32 image encoder
# *Hypothesis:* CLIP, trained contrastively on 400M (image, text) pairs, encodes *what the object is* (brand, type,
# text on packaging) more robustly than an ImageNet classifier → better matching across viewpoints/backgrounds.

# %%
run("Exp 2", C.CLIP_IMG, "CLIP ViT-B/32 (image)")

# %% [markdown]
# ### Experiment 3 — post-processing: PCA-whitening (fit on *train* split only)
# *Hypothesis:* raw CNN features are dominated by a few high-variance, correlated directions. Whitening to 256 dims
# equalises variance → better angular separation **and** 8× smaller index (computational win).

# %%
Er = get_embedding(df, C.RESNET, tag, normalize=False)
pca = PCA(n_components=min(256, len(rows["train"]) - 1, Er.shape[1]), whiten=True, random_state=0).fit(Er[rows["train"]])
Ew = l2norm(pca.transform(Er).astype(np.float32))
run("Exp 3", C.RESNET, f"ResNet-50 + PCA-whiten({Ew.shape[1]})", emb=("resnet_whiten", Ew), note="PCA fit on train only")

# %% [markdown]
# ### Experiment 4 — similarity measure (same raw ResNet-50 vectors)
# Cosine vs raw dot product vs (negative) Euclidean distance.

# %%
run("Exp 4a", C.RESNET, "ResNet-50 raw", "dot product", metric="dot", norm=False, emb=("resnet_raw_dot", Er))
run("Exp 4b", C.RESNET, "ResNet-50 raw", "euclidean (neg.)", metric="l2", norm=False, emb=("resnet_raw_l2", Er))

# %% [markdown]
# ### Results

# %%
R = pd.DataFrame(results); R.to_csv(RES / "experiments.csv", index=False)
show = R[["Experiment", "Model", "dim", "Similarity", "pair_thr", "AUC", "AP", "pair_F1", "retr_thr", "retr_F1", "cand_R", "embed_s", "emb_MB", "search_s(test)"]]
(RES / "experiments.md").write_text(md_table(show, "{:.3f}")); show

# %%
neg_cols = [c for c in R.columns if c.startswith("AUC_")]
print("AUC by negative type (test)"); R[["Model", "Similarity"] + neg_cols].round(3)

# %% [markdown]
# **Observations (edit after running):**
# * pHash vs learned embeddings – how much does semantic encoding add over near-duplicate detection?
# * Compare backbones on `AUC_image_hard_neg` (negatives that are pHash-close) – a high number means the embedding
#   understands more than pixel layout.
# * Whitening: did quality improve and by how much memory/time did we save?
# * Dot/L2 on raw vectors vs cosine: normalisation removes the influence of feature magnitude (image contrast,
#   texture density), which is irrelevant to identity → cosine is expected to win.
# * Compare with Part B: the image-only retrieval-F1 is typically lower than text-only; the two are complementary.

# %% [markdown]
# ## 4. Threshold analysis (best model, val → test)

# %%
# the downstream analysis uses cosine-type scores, so pick the best among cosine-based runs
best = R[R.metric == "cos"].sort_values("retr_F1", ascending=False).iloc[0]; print("best (cosine-based):", best.Model)
Eb = E[best.key]
sv = pair_scores(Eb, bv.i.values, bv.j.values); st_ = pair_scores(Eb, bt.i.values, bt.j.values)
thrs = np.linspace(np.quantile(sv, .02), np.quantile(sv, .98), 60); T = threshold_table(bv.y.values, sv, thrs)
t_star, _ = best_f1_threshold(bv.y.values, sv)
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
ax[0].plot(T.thr, T.precision, label="precision"); ax[0].plot(T.thr, T.recall, label="recall"); ax[0].plot(T.thr, T.f1, lw=2, label="F1")
ax[0].axvline(t_star, ls="--", c="k"); ax[0].legend(); ax[0].set_title("pair metrics vs threshold (val)"); ax[0].set_xlabel("cosine threshold")
ax[1].plot(T.thr, T.fp, c="tab:red", label="FP"); ax[1].plot(T.thr, T.fn, c="tab:blue", label="FN"); ax[1].axvline(t_star, ls="--", c="k")
ax[1].legend(); ax[1].set_title("error counts vs threshold (val)"); ax[1].set_xlabel("cosine threshold")
ax[2].plot(T.recall, T.precision); ax[2].set_title("precision-recall (val)"); ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision")
fig.tight_layout(); fig.savefig(RES / "c_threshold_analysis.png", dpi=110); plt.show()
pm = pair_metrics(bt.y.values, st_, t_star); print({k: round(v, 3) for k, v in pm.items()})

# %% [markdown]
# **Threshold selection:** maximise F1 on val and freeze it for test. Because image similarity saturates for
# *same-category* products (everything "looks like a bag"), the useful operating range is narrow and high; lowering it
# quickly floods the result with false positives (see FP curve).

# %% [markdown]
# ## 5. Nearest-neighbour analysis
# For test listings: embed → exact top-K search among test listings → check whether neighbours are truly the same product
# (green border = same `label_group`, red = different).

# %%
rt = rows["test"]; idx, scn = topk_search(Eb[rt], 10)
q_ok = np.where(gs[rt] >= 3)[0]; rng = np.random.RandomState(7); qs = rng.choice(q_ok, min(6, len(q_ok)), replace=False)
ids, border, caps = [], [], []
for q in qs:
    ids.append(rt[q]); border.append("royalblue"); caps.append(f"QUERY g{lab[rt[q]]}\n{df.title.iloc[rt[q]]}")
    for t in range(5):
        j = rt[idx[q, t]]; ids.append(j); border.append("green" if lab[j] == lab[rt[q]] else "red")
        caps.append(f"cos={scn[q, t]:.2f}\n{df.title.iloc[j]}")
image_grid(df, ids, captions=caps, ncols=6, border=border, size=2.2, title="Query (blue) and its top-5 visual neighbours (green = same product)",
           save=RES / "c_nearest_neighbours.png"); plt.show()

same = (lab[rt][idx] == lab[rt][:, None])
valid = gs[rt] >= 2
nn = {"hit@1": same[valid, 0].mean(), "precision@5": same[valid, :5].mean(),
      "recall@10": (same[valid, :10].sum(1) / np.minimum(gs[rt][valid] - 1, 10)).mean()}
print({k: round(float(v), 3) for k, v in nn.items()})
pd.DataFrame([nn]).to_csv(RES / "nn_metrics.csv", index=False)
# how does neighbour precision decay with rank?
fig, ax = plt.subplots(figsize=(5.5, 3.2)); ax.plot(range(1, 11), same[valid].mean(0), marker="o"); ax.set_xlabel("neighbour rank"); ax.set_ylabel("P(same product)")
ax.set_title("Precision of the k-th neighbour"); fig.tight_layout(); fig.savefig(RES / "c_rank_precision.png", dpi=110); plt.show()

# %% [markdown]
# **Reading the grid:** the top neighbours are often *visually* very similar yet belong to another product (red) — same
# category, same hero photo, different variant. Green neighbours are typically near-duplicates of the same photo. This
# is exactly the "embeddings capture appearance, not identity" limitation.

# %% [markdown]
# ## 6. Error analysis
# We tag each test pair with simple image-condition features and measure error rates per condition – this turns anecdotes
# into evidence for *why* failures occur.

# %%
sub_pairs = bt.sample(min(2500, len(bt)), random_state=0).copy(); sub_pairs["score"] = pair_scores(Eb, sub_pairs.i.values, sub_pairs.j.values)
sub_pairs["pred"] = (sub_pairs.score >= t_star).astype(int)
need = np.unique(np.r_[sub_pairs.i.values, sub_pairs.j.values]); feat = {}
for i in need:
    im = load_rgb(df.path.iloc[i]); a = np.asarray(im.convert("L"), dtype=np.float32)
    feat[i] = (im.width / im.height, a.std(), (a > 240).mean(), min(im.size))
F = lambda c, k: np.array([feat[x][k] for x in sub_pairs[c]])
sub_pairs["d_aspect"] = np.abs(F("i", 0) - F("j", 0)); sub_pairs["d_white"] = np.abs(F("i", 2) - F("j", 2))
sub_pairs["low_contrast"] = (np.minimum(F("i", 1), F("j", 1)) < 20); sub_pairs["small"] = (np.minimum(F("i", 3), F("j", 3)) < 200)
sub_pairs["phash_ham"] = pair_hamming(phash_u64(df.image_phash), sub_pairs.i.values, sub_pairs.j.values)
pos_ = sub_pairs[sub_pairs.y == 1]; neg_ = sub_pairs[sub_pairs.y == 0]
rows_out = []
for name, mask in [("all positives", pos_.y == 1), ("different aspect ratio (Δ>0.25)", pos_.d_aspect > .25), ("different background (Δ white-share>0.3)", pos_.d_white > .3),
                   ("low-contrast / washed-out image", pos_.low_contrast), ("low resolution (<200px)", pos_.small), ("phash far (>=24)", pos_.phash_ham >= 24), ("phash near (<=8)", pos_.phash_ham <= 8)]:
    d = pos_[mask]; rows_out.append({"condition (true matches)": name, "n": len(d), "missed (FN rate)": float((d.pred == 0).mean()) if len(d) else np.nan})
for name, mask in [("all negatives", neg_.y == 0), ("phash near (<=8) – visually same", neg_.phash_ham <= 8), ("hard negatives (text/image)", neg_.kind.isin(["text_hard_neg", "image_hard_neg"]))]:
    d = neg_[mask]; rows_out.append({"condition (true non-matches)": name, "n": len(d), "wrongly matched (FP rate)": float((d.pred == 1).mean()) if len(d) else np.nan})
err_tab = pd.DataFrame(rows_out); err_tab.to_csv(RES / "error_rates_by_condition.csv", index=False); err_tab

# %%
fn = sub_pairs[(sub_pairs.y == 1) & (sub_pairs.pred == 0)].sort_values("score"); fp = sub_pairs[(sub_pairs.y == 0) & (sub_pairs.pred == 1)].sort_values("score", ascending=False)
print(f"FN={len(fn)}  FP={len(fp)}")
if len(fn): pair_figure(df, list(zip(fn.i[:5], fn.j[:5])), [f"MISSED match\ncos={r.score:.2f}\nphash Ham={r.phash_ham}\nΔbg={r.d_white:.2f}" for r in fn.head(5).itertuples()], save=RES / "c_errors_false_negatives.png", title="False negatives: same product, low image similarity"); plt.show()
if len(fp): pair_figure(df, list(zip(fp.i[:5], fp.j[:5])), [f"WRONG match\ncos={r.score:.2f}\nphash Ham={r.phash_ham}\nkind={r.kind}" for r in fp.head(5).itertuples()], save=RES / "c_errors_false_positives.png", title="False positives: different products, high image similarity"); plt.show()

# %% [markdown]
# **Failure taxonomy (confirm/edit with the figures and the table above):**
#
# | Case | Observed behaviour | Reason |
# |---|---|---|
# | Same product photographed differently | FN | embedding encodes viewpoint/background; different crops change activations |
# | Different products, similar appearance | FP | ImageNet/CLIP features encode category & style, not SKU identity |
# | Different colours/variants | FP | colour is a small part of a global descriptor; hero image often reused |
# | Different packaging (box vs bare item) | FN | different visual content altogether |
# | Different backgrounds | FN | background pixels contribute to the global descriptor |
# | Cropped images | FN | partial object ⇒ different feature statistics |
# | Low-quality / washed-out | both | low-contrast images collapse to generic embeddings (everything is similar to everything) |
#
# **Fix directions:** fine-tune with metric learning (ArcFace / contrastive with hard negatives) on the train groups;
# multi-crop / TTA; background removal; and above all **fuse with text** (Finale).

# %% [markdown]
# ## 7. Computational considerations
# Exact all-pairs comparison is O(n²·d). For n = 34 250 listings that is ≈ 5.9 × 10⁸ distinct pairs; with d = 2048 float32
# the embedding matrix is 280 MB and a full similarity matrix would be 4.7 GB — so we (i) compute it **in chunks** and keep
# only the top-K, (ii) can shrink d with PCA (256-d ⇒ 8× less memory/time), and (iii) at larger scale would use an
# **approximate nearest-neighbour index** (HNSW/IVF, e.g. faiss). Embedding extraction (a forward pass per image) is the
# dominant one-off cost; `embed_s` in the table is measured per backbone.

# %%
try:
    import faiss
    X = np.ascontiguousarray(Eb[rt].astype("float32")); d = X.shape[1]
    t0 = time.perf_counter(); ex_idx, _ = topk_search(X, 10); t_exact = time.perf_counter() - t0
    hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hn.add(X)
    t0 = time.perf_counter(); _, ai = hn.search(X, 11); t_ann = time.perf_counter() - t0
    rec = np.mean([len(set(ex_idx[q]) & set(ai[q, 1:])) / 10 for q in range(len(X))])
    print(f"exact {t_exact:.2f}s vs HNSW {t_ann:.2f}s | recall@10 of HNSW vs exact = {rec:.3f}")
except ImportError:
    print("faiss not installed (optional): `pip install faiss-cpu` to run the exact-vs-ANN comparison")

# %% [markdown]
# ## 8. Conclusions & answers to the guiding questions
# * **What does an embedding capture?** global appearance/semantic category (shape, colour layout, texture, sometimes text),
#   not instance identity.
# * **Why can the same product differ?** viewpoint, background, crop, packaging, lighting; **why can different products be
#   close?** same category/hero image/variants.
# * **Best metric?** cosine on L2-normalised vectors (see Exp 4); whitening can help and shrinks the index.
# * **Threshold:** tuned on val; the usable range is narrow.
# * **Computation:** O(n²) → chunked top-K (exact) now, ANN at scale.
# * **Take-away:** images give a strong but imperfect signal that errs on *different* examples than text (see the
#   complementarity check in the Finale) → multimodal fusion.

# %%
json.dump({"best_model": best.Model, "pair_threshold": float(t_star), **{k: float(v) for k, v in nn.items()}}, open(RES / "summary.json", "w"), indent=2)
print("saved ->", RES)
