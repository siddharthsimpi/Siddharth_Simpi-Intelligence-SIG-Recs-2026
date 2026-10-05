# %% [markdown]
# # Part A — Dataset Exploration (Shopee Product Matching)
#
# **Goal:** understand the data *before* modelling. Every plot below is followed by what it tells us
# and what it implies for the model. Sections: (1) dataset structure, (2) statistics, (3) similarity
# case studies, (4) challenges, (5) answers to the guiding questions.
#
# > Run from inside `PartA/` (cwd is used to locate the shared `shopee_match` package).
# > Cells marked **Observations** contain my interpretation template — *re-read them against your own
# > outputs and edit anything that does not match what you see.*

# %%
import sys, re, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from shopee_match import config as C
from shopee_match.data import load_dataset, split_rows, get_tag
from shopee_match.text import normalize_title, numbers
from shopee_match.embeddings import get_embedding, load_rgb
from shopee_match.retrieval import topk_search, pair_scores
from shopee_match.pairs import build_benchmark_pairs, phash_u64, pair_hamming
from shopee_match.metrics import auc_ap
from shopee_match.viz import image_grid, pair_figure, hist_compare

RES = C.results_dir("PartA")
df = load_dataset()
tag = get_tag(df)
rows = split_rows(df)
summary = {}
print(f"{len(df):,} listings | data dir: {C.DATA_DIR}")

# %% [markdown]
# ## 1. Understanding the dataset
# ### 1.1 Files, columns and what they mean

# %%
cols = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "n_null": df.isna().sum(),
    "n_unique": df.nunique(),
    "example": df.iloc[0].astype(str).str.slice(0, 60),
})
meaning = {
    "posting_id": "unique id of one listing (a row)",
    "image": "image file name (MD5 of the file) -> identical files share a name",
    "image_phash": "64-bit perceptual hash of the image (hex); visually similar images -> small Hamming distance",
    "title": "free-text listing title written by the seller (noisy, multilingual)",
    "label_group": "ground truth: listings with the same value are the SAME product",
    "path": "(added) absolute path of the image", "split": "(added) train/val/test split, done BY label_group",
}
cols["meaning"] = pd.Series(meaning)
cols

# %% [markdown]
# **What constitutes a product group / how is matching represented?**
# `label_group` is an integer id: *two listings match iff they have the same `label_group`*. The matching
# relation is therefore an **equivalence relation** (reflexive, symmetric, transitive) – the data is a
# partition of listings into clusters. The Kaggle task asks, for each listing, to output the set of all
# listings in its cluster (always including itself); the metric is the mean per-row F1 of that set.
# Only `train.csv` is labelled (the public `test.csv` has 3 dummy rows), so *we create our own
# train/val/test split — grouped by `label_group`* so no product leaks across splits.

# %% [markdown]
# ## 2. Statistical analysis
# ### 2.1 Listings, groups and group sizes

# %%
gsz = df.groupby("label_group").size()
size_dist = gsz.value_counts().sort_index()
summary.update(n_listings=len(df), n_groups=int(gsz.size), mean_group_size=float(gsz.mean()),
               median_group_size=float(gsz.median()), max_group_size=int(gsz.max()),
               share_groups_size2=float((gsz == 2).mean()), share_singletons=float((gsz == 1).mean()))
print(f"listings={len(df):,}  groups={gsz.size:,}  mean size={gsz.mean():.2f}  median={gsz.median():.0f}  max={gsz.max()}")
print(f"groups of size 1: {(gsz==1).mean():.1%} | size 2: {(gsz==2).mean():.1%} | size>=5: {(gsz>=5).mean():.1%}")
print(f"-> a random listing has on average {(gsz*gsz).sum()/gsz.sum()-1:.1f} true matches")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].bar(size_dist.index, size_dist.values); ax[0].set_yscale("log")
ax[0].set_xlabel("group size (#listings of the same product)"); ax[0].set_ylabel("#groups (log)"); ax[0].set_title("Distribution of group sizes")
ax[1].hist(gsz.values, bins=np.arange(1, min(gsz.max(), 60) + 2) - 0.5, cumulative=True, density=True)
ax[1].set_xlabel("group size"); ax[1].set_title("CDF of group size"); fig.tight_layout()
fig.savefig(RES / "a_group_sizes.png", dpi=110); plt.show()

# %% [markdown]
# **Observations (edit):** group sizes are heavily right-skewed: most products have very few listings while a
# handful have dozens. Consequences: (i) the *average* listing has few true matches, so a model that
# over-predicts matches is punished heavily by the per-row F1; (ii) large groups make **recall** hard
# (many different phrasings of the same product); (iii) a fixed "top-k" must be ≥ the max group size to
# allow perfect recall.

# %% [markdown]
# ### 2.2 Images: unique files, duplicates and perceptual hashes

# %%
n_img, n_ph = df["image"].nunique(), df["image_phash"].nunique()
dup_img_rows = df["image"].duplicated(keep=False).sum()
g_per_img = df.groupby("image")["label_group"].nunique()
g_per_ph = df.groupby("image_phash")["label_group"].nunique()
rows_ph_conflict = df["image_phash"].map(g_per_ph).gt(1).sum()
grp_one_phash = (df.groupby("label_group")["image_phash"].nunique() == 1).mean()
grp_all_same_img = (df.groupby("label_group")["image"].nunique() == 1).mean()
summary.update(unique_images=int(n_img), unique_phash=int(n_ph), rows_with_duplicate_image=int(dup_img_rows),
               images_in_multiple_groups=int((g_per_img > 1).sum()), phash_in_multiple_groups=int((g_per_ph > 1).sum()),
               groups_with_single_phash=float(grp_one_phash))
print(f"unique image files: {n_img:,} / {len(df):,} rows  ({dup_img_rows:,} rows share an image file with another row)")
print(f"unique phash: {n_ph:,}")
print(f"image files that appear in >1 product groups: {(g_per_img>1).sum():,}   <- identical picture, DIFFERENT product (label noise / seller re-use)")
print(f"phash values that appear in >1 product groups: {(g_per_ph>1).sum():,} ({rows_ph_conflict:,} rows)")
print(f"groups whose listings all share ONE phash: {grp_one_phash:.1%};  all share the same image file: {grp_all_same_img:.1%}")

# %% [markdown]
# ### 2.3 Titles: duplicates, similarity, length, language and noise

# %%
df["title_norm"] = df["title"].astype(str).map(normalize_title)
df["n_chars"] = df["title"].astype(str).str.len()
df["n_words"] = df["title_norm"].str.split().str.len()
exact_dup = df["title"].duplicated(keep=False)
norm_dup = df["title_norm"].duplicated(keep=False)
g_per_title = df.groupby("title_norm")["label_group"].nunique()
summary.update(share_rows_exact_dup_title=float(exact_dup.mean()), share_rows_norm_dup_title=float(norm_dup.mean()),
               titles_in_multiple_groups=int((g_per_title > 1).sum()))
print(f"rows whose raw title is duplicated elsewhere: {exact_dup.mean():.1%}; after normalisation: {norm_dup.mean():.1%}")
print(f"normalised titles shared by >1 product group: {(g_per_title>1).sum():,}  <- identical title, different product")
print(df[["n_chars", "n_words"]].describe(percentiles=[.05, .5, .95]).round(1))

# language / script / noise heuristics
ID_MARK = set("dan untuk yang dengan murah promo original baju tas sepatu anak bayi wanita pria ready stok gratis bisa buat kode warna ukuran paket isi".split())
EN_MARK = set("the for and with new free set pack cotton men women baby kids girls boys size color black white blue".split())
def lang_bucket(t):
    toks = set(t.split())
    if re.search(r"[\u0E00-\u0E7F\u3040-\u30ff\u4e00-\u9fff\u0600-\u06ff\uac00-\ud7af]", t): return "non-latin script"
    a, b = len(toks & ID_MARK) > 0, len(toks & EN_MARK) > 0
    return "both" if a and b else "indonesian/malay markers" if a else "english markers" if b else "no marker (brand/model/other)"
df["lang"] = df["title_norm"].map(lang_bucket)
df["allcaps"] = df["title"].astype(str).map(lambda s: s.isupper())
df["has_digit"] = df["title"].astype(str).str.contains(r"\d")
df["special_ratio"] = df["title"].astype(str).map(lambda s: sum(not (c.isalnum() or c.isspace()) for c in s) / max(len(s), 1))
df["non_ascii"] = df["title"].astype(str).map(lambda s: any(ord(c) > 127 for c in s))

fig, ax = plt.subplots(1, 3, figsize=(15, 3.5))
ax[0].hist(df.n_chars, bins=50); ax[0].set_title("Title length (characters)")
ax[1].hist(df.n_words, bins=np.arange(0, 40) - 0.5); ax[1].set_title("Title length (words)")
df["lang"].value_counts().plot.barh(ax=ax[2]); ax[2].set_title("Language / script heuristic")
fig.tight_layout(); fig.savefig(RES / "a_title_stats.png", dpi=110); plt.show()
print(f"ALL-CAPS titles: {df.allcaps.mean():.1%} | contain digits: {df.has_digit.mean():.1%} | non-ASCII chars: {df.non_ascii.mean():.1%} | >=15% punctuation/emoji: {(df.special_ratio>.15).mean():.1%} | <=3 words: {(df.n_words<=3).mean():.1%}")
summary.update(share_allcaps=float(df.allcaps.mean()), share_non_ascii=float(df.non_ascii.mean()), share_short_titles=float((df.n_words <= 3).mean()))

# %% [markdown]
# **Observations (edit):** titles are long, seller-written strings stuffed with keywords, promo words,
# sizes and brand names; a large part is Indonesian/Malay with English mixed in (code-switching). A
# non-trivial share contains emoji/special symbols or ALL CAPS. Title length varies a lot (very short
# titles carry little information; very long ones carry irrelevant keyword spam).

# %% [markdown]
# ### 2.4 Image properties (sampled)

# %%
rng = np.random.RandomState(0)
samp = rng.choice(len(df), min(1500, len(df)), replace=False)
stats = []
for i in samp:
    im = load_rgb(df.path.iloc[i]); a = np.asarray(im.convert("L"), dtype=np.float32)
    stats.append((im.width, im.height, im.width / im.height, a.mean(), a.std(), (a > 240).mean()))
st = pd.DataFrame(stats, columns=["w", "h", "aspect", "brightness", "contrast", "white_share"])
print(st.describe().round(2))
fig, ax = plt.subplots(1, 3, figsize=(14, 3.2))
ax[0].hist(st.aspect.clip(0, 3), bins=40); ax[0].set_title("Aspect ratio (w/h)")
ax[1].hist(st.contrast, bins=40); ax[1].set_title("Contrast (grey std) – low = washed out/blank")
ax[2].hist(st.white_share, bins=40); ax[2].set_title("Share of near-white pixels (studio backgrounds)")
fig.tight_layout(); fig.savefig(RES / "a_image_stats.png", dpi=110); plt.show()

# %% [markdown]
# ## 3. Product-similarity case studies
# To keep this fast we analyse the **val split** (~20 % of the groups; the groups are complete).
# First, quantify *how separable* matches are with the two cheapest signals available in the CSV —
# title similarity (char n-gram TF-IDF) and image perceptual hash.

# %%
v = rows["val"]
bench_all = build_benchmark_pairs(df, v, max_pos=15000)
Ech = get_embedding(df, "tfidf_char", tag)
ph = phash_u64(df["image_phash"])
bench_all["text_cos"] = pair_scores(Ech, bench_all.i.values, bench_all.j.values)
bench_all["ham"] = pair_hamming(ph, bench_all.i.values, bench_all.j.values)
pos, neg = bench_all[bench_all.y == 1], bench_all[bench_all.y == 0]
print(bench_all.kind.value_counts().to_dict())
print("\nText char-TF-IDF cosine:  AUC(all)=%.3f  AUC(vs random)=%.3f  AUC(vs text-hard)=%.3f" % (
    auc_ap(bench_all.y, bench_all.text_cos)[0],
    auc_ap(*(lambda m: (bench_all.y[m], bench_all.text_cos[m]))(bench_all.kind.isin(["positive", "random_neg"])))[0],
    auc_ap(*(lambda m: (bench_all.y[m], bench_all.text_cos[m]))(bench_all.kind.isin(["positive", "text_hard_neg"])))[0]))
print("phash (neg. Hamming):     AUC(all)=%.3f  AUC(vs image-hard)=%.3f" % (
    auc_ap(bench_all.y, -bench_all.ham)[0],
    auc_ap(*(lambda m: (bench_all.y[m], -bench_all.ham[m]))(bench_all.kind.isin(["positive", "image_hard_neg"])))[0]))
print(f"\npositive pairs with IDENTICAL phash: {(pos.ham==0).mean():.1%} | Hamming<=8: {(pos.ham<=8).mean():.1%} | >=24: {(pos.ham>=24).mean():.1%}")
print(f"negative pairs with IDENTICAL phash: {(neg.ham==0).mean():.2%}  (these are exact image duplicates of DIFFERENT products)")
summary.update(pos_pairs_same_phash=float((pos.ham == 0).mean()), pos_pairs_far_phash=float((pos.ham >= 24).mean()))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
for k, c in [("positive", "tab:green"), ("random_neg", "tab:gray"), ("text_hard_neg", "tab:red")]:
    ax[0].hist(bench_all[bench_all.kind == k].text_cos, bins=40, alpha=.5, density=True, label=k, color=c)
ax[0].set_xlabel("title char-TF-IDF cosine"); ax[0].legend(); ax[0].set_title("Title similarity: matches vs non-matches")
for k, c in [("positive", "tab:green"), ("random_neg", "tab:gray"), ("image_hard_neg", "tab:red")]:
    ax[1].hist(bench_all[bench_all.kind == k].ham, bins=np.arange(0, 66, 2), alpha=.5, density=True, label=k, color=c)
ax[1].set_xlabel("phash Hamming distance"); ax[1].legend(); ax[1].set_title("Image-hash distance: matches vs non-matches")
fig.tight_layout(); fig.savefig(RES / "a_pair_separability.png", dpi=110); plt.show()

# %% [markdown]
# **Observations (edit):** random non-matches are easy to reject on either signal, but the *hard* negatives
# (similar title, or near-identical image, but different product) overlap strongly with the positives.
# Identical-phash positives are plentiful (cheap high-precision rule) yet a sizeable minority of true
# matches have completely different pictures – images alone cannot reach full recall, and titles alone
# cannot separate variants.

# %% [markdown]
# ### 3.1 Same product, multiple listings

# %%
big = gsz[(gsz >= 4) & (gsz <= 6)].index
sel_groups = pd.Series(big).sample(3, random_state=1).tolist()
ids = [i for g in sel_groups for i in np.where(df.label_group.values == g)[0][:6]]
fig = image_grid(df, ids, captions=[f"g{df.label_group.iloc[i]}: {df.title.iloc[i]}" for i in ids], ncols=6,
                 title="Three products, each with several listings", save=RES / "a_ex_same_product.png"); plt.show()

# %% [markdown]
# **Why hard:** same product, yet titles differ in word order, keywords, sizes, language and
# promotional words, and pictures differ in angle/background/crop. A model must learn *what is invariant*
# (brand, product type, model, size) versus *what is seller-specific noise*.

# %% [markdown]
# ### 3.2 Same product, different images

# %%
sub = df.iloc[v]
G = []
for g, idx in sub.groupby("label_group").indices.items():
    if len(idx) < 2: continue
    gi = v[idx]; h = pair_hamming(ph, np.repeat(gi, len(gi)), np.tile(gi, len(gi))).reshape(len(gi), len(gi))
    a, b = np.unravel_index(h.argmax(), h.shape)
    G.append((h[a, b], gi[a], gi[b]))
G = sorted(G, reverse=True)[:5]
pairs = [(a, b) for _, a, b in G]
caps = [f"group {df.label_group.iloc[a]}\nphash Hamming={h}\ntitle cos={pair_scores(Ech, np.array([a]), np.array([b]))[0]:.2f}" for h, a, b in G]
pair_figure(df, pairs, caps, save=RES / "a_ex_same_product_diff_image.png", title="Same product, very different images"); plt.show()

# %% [markdown]
# **Why hard:** different viewpoint, packaging shot vs. product shot, lifestyle photo vs. white
# background, collage vs. single item. Pixel-level/perceptual hashes fail; a semantic image encoder
# helps only partially, so *text becomes the necessary complementary signal* here.

# %% [markdown]
# ### 3.3 Similar titles, different products

# %%
Xv = Ech[v]
idx_v, sc_v = topk_search(Xv, 5)
A, B, S = np.repeat(v, idx_v.shape[1]), v[idx_v].ravel(), sc_v.ravel()
m = (df.label_group.values[A] != df.label_group.values[B]) & (A < B)
cand = pd.DataFrame({"i": A[m], "j": B[m], "cos": S[m]}).sort_values("cos", ascending=False).drop_duplicates(["i"]).head(5)
caps = [f"DIFFERENT groups\n{df.label_group.iloc[r.i]} vs {df.label_group.iloc[r.j]}\ntitle cos={r.cos:.2f}\nphash Ham={pair_hamming(ph, np.array([r.i]), np.array([r.j]))[0]}" for r in cand.itertuples()]
pair_figure(df, list(zip(cand.i, cand.j)), caps, save=RES / "a_ex_similar_title_diff_product.png", title="Near-identical titles, different products"); plt.show()

# %% [markdown]
# **Why hard:** titles differ only in a *discriminative detail* (colour, size, model number, pack size,
# variant) which a bag-of-words/semantic embedding treats as a tiny difference, while it defines product identity.
# Number tokens and colours need special attention; the image can help (colour) – or not (size).

# %% [markdown]
# ### 3.4 Visually near-identical images, different products

# %%
negs = bench_all[(bench_all.y == 0) & (bench_all.ham <= 2)].sort_values("text_cos").head(5)
if len(negs):
    caps = [f"DIFFERENT groups\nphash Ham={r.ham}\ntitle cos={r.text_cos:.2f}" for r in negs.itertuples()]
    pair_figure(df, list(zip(negs.i, negs.j)), caps, save=RES / "a_ex_similar_image_diff_product.png",
                title="Near-identical images, different products"); plt.show()
else:
    print("no such negative pair in this split")

# %% [markdown]
# **Why hard:** sellers reuse stock photos / product category templates, or sell different variants
# with the same hero image. Image similarity alone gives *false positives* – and some of these may even be
# label noise in the dataset (worth stating in the report).

# %% [markdown]
# ### 3.5 Noisy / incomplete listings

# %%
noisy = df.iloc[v]
cases = {
    "very short title (<=2 words)": noisy[noisy.n_words <= 2],
    "ALL CAPS": noisy[noisy.allcaps & (noisy.n_words > 3)],
    "many symbols / emoji": noisy[noisy.special_ratio > 0.15],
    "non-latin script": noisy[noisy.lang == "non-latin script"],
}
for name, d in cases.items():
    print(f"\n### {name}: {len(d)} rows in val")
    print(d.title.head(4).to_string(index=False))
ids = [d.index[0] for d in cases.values() if len(d)]
if ids:
    image_grid(df, ids, ncols=len(ids), save=RES / "a_ex_noisy.png", title="Noisy / incomplete listings"); plt.show()

# titles of the SAME product with the lowest title similarity inside the group
worst = []
for g, idx in sub.groupby("label_group").indices.items():
    if len(idx) >= 3:
        gi = v[idx]; S = (Ech[gi] @ Ech[gi].T).toarray(); np.fill_diagonal(S, 1); a, b = np.unravel_index(S.argmin(), S.shape)
        worst.append((S[a, b], gi[a], gi[b]))
worst = sorted(worst)[:4]
pair_figure(df, [(a, b) for _, a, b in worst], [f"same product\ntitle cos={s:.2f}" for s, _, _ in worst],
            save=RES / "a_ex_same_product_diff_title.png", title="Same product, almost no title overlap"); plt.show()

# %% [markdown]
# ### 3.6 Number tokens as variant markers

# %%
nums = df["title_norm"].map(numbers)
within = []
for g, idx in sub.groupby("label_group").indices.items():
    gi = v[idx]
    for a in range(len(gi)):
        for b in range(a + 1, len(gi)):
            na, nb = nums.iloc[gi[a]], nums.iloc[gi[b]]
            if na and nb: within.append(na == nb)
print(f"positive pairs where both titles contain numbers: {len(within):,}; number sets identical in {np.mean(within):.1%}")
neg_n = [(nums.iloc[a], nums.iloc[b]) for a, b in zip(bench_all[bench_all.kind == 'text_hard_neg'].i, bench_all[bench_all.kind == 'text_hard_neg'].j)]
neg_n = [x == y for x, y in neg_n if x and y]
print(f"text-hard NEGATIVE pairs where both contain numbers: {len(neg_n):,}; number sets identical in {np.mean(neg_n):.1%}")
print("-> if these two rates differ a lot, 'number tokens differ' is a strong negative cue we can feed to the Finale model")

# %% [markdown]
# ## 4. Challenges identified
#
# | # | Challenge | Evidence in this notebook | Implication |
# |---|-----------|---------------------------|-------------|
# | 1 | **Noisy, keyword-stuffed titles** (promo words, caps, emoji) | §2.3, §3.5 | normalisation, char n-grams, robust encoders |
# | 2 | **Multilingual / code-switching** (Indonesian, Malay, English, other scripts) | §2.3 | multilingual sentence encoder; char n-grams |
# | 3 | **Abbreviations / spelling variants / no standard word order** | §3.1, §3.3 | sub-word / char features beat exact-word matching |
# | 4 | **Same product, different photos** (angle, background, crop, packaging) | §3.2 | image alone caps recall → need text |
# | 5 | **Different products, same / similar photo** (stock photos, variants) | §3.4, §2.2 | image alone hurts precision; exact-image duplicates can disagree on label |
# | 6 | **Variants hidden in small details** (size, colour, model number) | §3.3, §3.6 | number/variant features; learned fusion |
# | 7 | **Highly skewed group sizes** | §2.1 | per-row F1 punishes over-matching; thresholds must be calibrated |
# | 8 | **Very short / uninformative titles or broken images** | §3.5, §2.4 | missing-modality robustness; fall back to the other modality |
# | 9 | **Scale**: O(n²) pair comparisons | n ≈ 34k → 1.2 × 10⁹ pairs | blocking / top-k retrieval / ANN, never classify all pairs |
# | 10 | **Possible label noise** (identical image+title in different groups) | §2.2, §2.3 | upper bound on achievable F1; interpret errors carefully |
#
# ## 5. Answers to the guiding questions
# **What makes two listings belong to the same product?** — Being the same *sellable item identity*
# (brand + model/type + size/variant), regardless of seller wording or photo. It is **not** surface similarity.
#
# **Can titles alone decide?** — Largely, yes for easy cases (see AUC above) but not reliably: variants
# share almost the whole title (false positives) and the same product can have disjoint titles (false negatives).
#
# **Can images alone decide?** — Same issue from the other side: identical hero photos for different
# variants (FP) and very different photos for the same product (FN).
#
# **What is difficult for a model?** — Variants, hard negatives with near-identical titles/images,
# disjoint titles, large groups, very short titles.
#
# **What else is useful?** — `image_phash` (cheap exact-duplicate signal, Hamming distance), number tokens,
# title length, group-size prior, and *transitivity* of the matching relation (cluster-level reasoning).
#
# *(Edit these answers with the exact numbers printed above before you submit.)*

# %%
json.dump(summary, open(RES / "dataset_summary.json", "w"), indent=2)
pd.DataFrame({"group_size": size_dist.index, "n_groups": size_dist.values}).to_csv(RES / "group_size_distribution.csv", index=False)
print("saved results to", RES)
