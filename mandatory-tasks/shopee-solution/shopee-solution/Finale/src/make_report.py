"""Build Finale/report.pdf from the artefacts the notebook saved in Finale/results/.
All numbers come from results/final_metrics.json - nothing is typed by hand, so report and code cannot drift apart.

    python Finale/src/make_report.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PILImage

RES = HERE / "results"
_MAP = {"→": "->", "≥": ">=", "≤": "<=", "⊕": "+", "Δ": "delta ", "×": "x", "·": "*", "−": "-", "✓": "yes", "✗": "no", "–": "-", "—": "-",
        "≈": "~", "…": "...", "‘": "'", "’": "'", "“": '"', "”": '"', "τ": "thr", "≠": "!=", "√": "sqrt", "∈": "in", "²": "^2"}


def clean(x):
    s = str(x)
    for k, v in _MAP.items():
        s = s.replace(k, v)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.encode("latin-1", "replace").decode("latin-1")


def f4(x):
    return f"{x:.4f}" if isinstance(x, float) else clean(x)


def build(out=None):
    out = Path(out or HERE / "report.pdf")
    M = json.load(open(RES / "final_metrics.json"))
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.2, leading=12.2)
    SM = ParagraphStyle("SM", parent=B, fontSize=7.6, leading=9.4)
    story = []
    P = lambda t, st=B: story.append(Paragraph(clean(t) if "<b>" not in t else t, st))

    def table(df, widths=None, fs=7.6):
        data = [[Paragraph(f"<b>{clean(c)}</b>", SM) for c in df.columns]] + [[Paragraph(f4(v), SM) for v in r] for r in df.values]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.extend([t, Spacer(1, 6)])

    def img(name, w=15.5 * cm, max_h=21 * cm):
        """Add a figure; figures taller than a page are cut into page-sized horizontal slices (no layout error)."""
        p = RES / name
        if not p.exists():
            return
        im = PILImage.open(p).convert("RGB")
        iw, ih = im.size
        total_h = w * ih / iw
        k = max(1, int(-(-total_h // max_h)))               # ceil
        step = -(-ih // k)
        for n in range(k):
            top, bot = n * step, min(ih, (n + 1) * step)
            part = im if k == 1 else im.crop((0, top, iw, bot))
            if k > 1:
                p_part = RES / f"_part_{Path(name).stem}_{n}.png"
                part.save(p_part)
            else:
                p_part = p
            story.append(Image(str(p_part), width=w, height=w * (bot - top) / iw))
        story.append(Spacer(1, 6))

    d, fin, base = M["dataset"], M["final"], M["baseline"]
    story.append(Paragraph("Shopee Product Matching - Final Report (Multimodal)", ss["Title"]))
    P(f"Listings: {d['n']:,} (train {d['n_train']:,} / val {d['n_val']:,} / test {d['n_test']:,}); split by product group, thresholds tuned on val, all numbers on test. "
      f"Metric: Kaggle-style mean per-listing F1 over predicted match sets (\"retrieval-F1\").")
    if d.get("smoke"):
        P("<b>WARNING: these numbers were produced in SMOKE mode on synthetic stand-in data/models and only verify that the pipeline runs. "
          "Re-run on the real Kaggle data to obtain meaningful results.</b>", B)

    # 1 Approach
    P("1. Approach", H1)
    P(
      clean(f"Two-stage retrieve-and-rerank system. Stage 1: every listing retrieves its top-{M['candidate']['k']} neighbours under each of 4 encoders ({', '.join(M['candidate']['mods'])}); the union forms the candidate pool "
            f"(avoids O(n^2) classification; candidate-recall ceiling on test = {M['candidate']['recall_test']:.3f}). Stage 2: each candidate pair is described by 16 features (cosines of 3 text and 2 image encoders, "
            "perceptual-hash distance/equality/same-file, title length ratio, word Jaccard, number-token Jaccard and conflict flags) and scored by weighted score fusion or a gradient-boosted classifier. "
            "A threshold chosen on val turns scores into matches; optional graph post-processing (symmetry / connected components) uses the fact that matching is an equivalence relation."))
    P(f"<b>Final system (selected on val):</b> scorer = {clean(fin['scorer'])}; post-processing = {fin['post']}; threshold = {fin['thr']:.4f}. "
      f"<b>Test retrieval-F1 = {fin['test_F1']:.4f}</b> (precision {fin['test_P']:.4f}, recall {fin['test_R']:.4f}).")
    P("Baseline: equal-weight score fusion of the best single text cosine and the best single image cosine, cosine similarity, no learning, K=20 per encoder, threshold tuned on val. "
      f"Baseline test F1 = {base['test_F1']:.4f}.")

    # 2 Experiments
    P("2. Experiments", H1)
    S = pd.DataFrame(M["singles"])[["method", "modality", "val_F1", "test_F1", "test_P", "test_R"]]
    P("Single-modality systems (same candidate pool):", B); table(S)
    for e in M["experiments"]:
        P(f"{e['id']} - {e['title']}", H2)
        P(f"<b>Hypothesis.</b> {clean(e['hypothesis'])}<br/><b>Experiment.</b> {clean(e['experiment'])}<br/><b>Result.</b> {clean(e['result'])} -> <b>{clean(e['verdict'])}</b><br/>"
          f"<b>Analysis.</b> {clean(e['analysis'])}<br/><b>Conclusion.</b> {clean(e['conclusion'])}")
    img("final_e1_weight_sweep.png", 9 * cm)
    ex = pd.read_csv(RES / "experiments.csv"); table(ex)

    # 3 Analysis
    P("3. Analysis", H1)
    best_single = S.sort_values("test_F1").iloc[-1]
    ctx = M.get("context", {})
    gain = fin["test_F1"] - best_single["test_F1"]
    txt = (f"The best single encoder is {best_single['method']} ({best_single['test_F1']:.4f}); the final multimodal system reaches {fin['test_F1']:.4f} "
           f"({gain:+.4f} absolute). ")
    if "partB_best_retr_F1" in ctx and "partC_best_retr_F1" in ctx:
        txt += f"For reference, the best text-only system of Part B reached {ctx['partB_best_retr_F1']:.4f} and the best image-only system of Part C {ctx['partC_best_retr_F1']:.4f} (full-split top-50 retrieval). "
    txt += ("Experiments marked SUPPORTED above are those where the hypothesis held on the test split; for REJECTED/NOT SUPPORTED ones the extra complexity did not pay off, "
            "which is itself a finding: components are only kept in the final system if they improved val F1. Text and image encoders err on different pairs (see error categories), "
            "which is why fusion helps; side-information features mostly act on false positives (variants with conflicting numbers, re-used pictures).")
    P(txt)

    # 4 Error analysis
    P("4. Error analysis", H1)
    er = M["errors"]
    P(f"Stage-1 misses: {er['missed_by_retrieval']:,} of {er['true_pairs']:,} true directed pairs ({er['missed_by_retrieval'] / max(er['true_pairs'], 1):.1%}) never enter the candidate pool and cannot be recovered.")
    ec = pd.read_csv(RES / "error_why_fix.csv") if (RES / "error_why_fix.csv").exists() else pd.DataFrame()
    if len(ec):
        table(ec, widths=[5.2 * cm, 1.4 * cm, 5 * cm, 5.4 * cm])
    sl = pd.read_csv(RES / "error_slices.csv") if (RES / "error_slices.csv").exists() else pd.DataFrame()
    if len(sl):
        P("Error rates for interpretable slices:", B); table(sl)
    img("final_errors_fp.png", 12 * cm); img("final_errors_fn.png", 12 * cm)

    # 5 Ablation
    P("5. Ablation study", H1)
    A = pd.read_csv(RES / "ablation.csv"); table(A)
    img("final_ablation.png", 11 * cm)

    # 6/7
    P("6. Limitations", H1)
    P("Frozen pretrained encoders (no fine-tuning for product identity); fixed top-K candidate pool bounds recall; a single global threshold regardless of group size; "
      "TF-IDF is fitted on the evaluated corpus (unsupervised, transductive); possible label noise; results from one random group split without confidence intervals; "
      "no explicit colour/size/brand attribute extraction.")
    P("7. Future improvements", H1)
    P("Metric-learning fine-tuning (ArcFace/contrastive with hard-negative mining) of image and text encoders; joint multimodal embeddings; attribute extraction with conflict rules; "
      "HNSW index for scale; database-side augmentation / query expansion; adaptive thresholds per neighbourhood; cross-validated group splits with confidence intervals; community detection instead of plain connected components.")
    P("8. External resources", H1)
    P("scikit-learn (TF-IDF, HistGradientBoosting); sentence-transformers paraphrase-multilingual-MiniLM-L12-v2 (Reimers & Gurevych, 2019); OpenAI CLIP ViT-B/32 via HuggingFace transformers (Radford et al., 2021); "
      "torchvision ResNet-50 / EfficientNet-B0 (ImageNet); Kaggle 'Shopee - Price Match Guarantee' dataset and its F1 metric; perceptual hashes provided in the dataset.")

    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                      title="Shopee Product Matching - Final Report").build(story)
    return out


if __name__ == "__main__":
    print("wrote", build())
