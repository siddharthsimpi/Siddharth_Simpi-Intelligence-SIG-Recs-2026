"""Evaluation metrics implemented from scratch: BLEU, ROUGE-1/2/L, distinct-n, grounding overlap, repetition."""
from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(refs, hyps, max_n: int = 4, smooth: bool = True) -> dict:
    """Corpus BLEU (Papineni et al. 2002) with one reference per hypothesis (token lists).
    smooth=True applies add-1 smoothing to the n>1 precisions (Lin & Och 2004) so short/weak outputs do not collapse to 0."""
    match = [0] * max_n
    total = [0] * max_n
    hyp_len = ref_len = 0
    for r, h in zip(refs, hyps):
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hn, rn = _ngrams(h, n), _ngrams(r, n)
            match[n - 1] += sum(min(c, rn[g]) for g, c in hn.items())
            total[n - 1] += max(len(h) - n + 1, 0)
    precisions = []
    for n in range(max_n):
        if smooth and n > 0:
            precisions.append((match[n] + 1) / (total[n] + 1))
        else:
            precisions.append(match[n] / total[n] if total[n] > 0 else 0.0)
    if min(precisions) <= 0 or hyp_len == 0:
        bleu = 0.0
    else:
        bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if hyp_len > ref_len or hyp_len == 0 else math.exp(1 - ref_len / max(hyp_len, 1))
    return {"bleu": 100 * bleu, "precisions": [100 * p for p in precisions], "bp": bp,
            "hyp_len": hyp_len, "ref_len": ref_len}


def sentence_bleu(ref, hyp, max_n: int = 4) -> float:
    return corpus_bleu([ref], [hyp], max_n, smooth=True)["bleu"]


def _f1(match, n_hyp, n_ref):
    if match == 0 or n_hyp == 0 or n_ref == 0:
        return 0.0
    p, r = match / n_hyp, match / n_ref
    return 2 * p * r / (p + r)


def rouge_n(ref, hyp, n: int) -> float:
    rn, hn = _ngrams(ref, n), _ngrams(hyp, n)
    match = sum(min(c, rn[g]) for g, c in hn.items())
    return _f1(match, sum(hn.values()), sum(rn.values()))


def _lcs(a, b) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(ref, hyp) -> float:
    return _f1(_lcs(ref, hyp), len(hyp), len(ref))


def rouge_scores(refs, hyps) -> dict:
    """Mean sentence-level ROUGE-1/2/L F1 (x100)."""
    n = max(len(refs), 1)
    return {"rouge1": 100 * sum(rouge_n(r, h, 1) for r, h in zip(refs, hyps)) / n,
            "rouge2": 100 * sum(rouge_n(r, h, 2) for r, h in zip(refs, hyps)) / n,
            "rougeL": 100 * sum(rouge_l(r, h) for r, h in zip(refs, hyps)) / n}


def distinct_n(hyps, n: int) -> float:
    """Fraction of unique n-grams among all generated n-grams (diversity)."""
    grams = [g for h in hyps for g in _ngrams(h, n).elements()]
    return len(set(grams)) / max(len(grams), 1)


def repetition_rate(hyps) -> float:
    """Share of generated tokens that repeat the immediately previous token."""
    rep = tot = 0
    for h in hyps:
        for a, b in zip(h, h[1:]):
            rep += a == b
            tot += 1
    return rep / max(tot, 1)


STOP = set("the a an of and or to in on at is are was were be been it its this that these those for with as by from "
           "he she they we you i me my your his her their our not no yes so but if then than too very can will just "
           "hai hain tha thi the ho hoga hogi to ki ke ka ko se me mein par aur ya nahi nahin haan kya ye yeh wo woh "
           "bhi hi ek is us un in kuch bahut mujhe tumhe aap main tum hum".split())


STOP_DEV = set("\u0939\u0948 \u0939\u0948\u0902 \u0925\u093e \u0925\u0940 \u0925\u0947 \u0939\u094b \u0939\u094b\u0917\u093e \u0939\u094b\u0917\u0940 \u0915\u0940 \u0915\u0947 \u0915\u093e \u0915\u094b \u0938\u0947 \u092e\u0947\u0902 \u092a\u0930 \u0914\u0930 \u092f\u093e \u0928\u0939\u0940\u0902 \u0939\u093e\u0901 \u0939\u093e\u0902 \u0915\u094d\u092f\u093e \u092f\u0947 \u092f\u0939 \u0935\u094b \u0935\u0939 \u092d\u0940 \u0939\u0940 \u090f\u0915 \u0907\u0938 \u0909\u0938 \u0915\u0941\u091b \u092c\u0939\u0941\u0924 \u092e\u0941\u091d\u0947 \u0924\u0941\u092e\u094d\u0939\u0947\u0902 \u0906\u092a \u092e\u0948\u0902 \u0924\u0941\u092e \u0939\u092e \u0935\u0947 \u0924\u094b \u0915\u093f \u0925\u0940\u0902 \u0939\u0942\u0901 \u0939\u0942\u0902 \u0932\u093f\u090f \u0938\u093e\u0925 \u0915\u0930 \u0930\u0939\u093e \u0930\u0939\u0940 \u0930\u0939\u0947 \u091c\u093e".split())


def content_words(tokens, min_len: int = 4, min_len_dev: int = 3):
    """Content words of a token list: Latin words with >= min_len characters, Devanagari words with >= min_len_dev code points
    (Hindi words are written with fewer, combining, characters), no stop words, no numbers/punctuation/tags."""
    from .tokenize import is_wordlike, script_of
    out = []
    for t in tokens:
        if not is_wordlike(t) or t in STOP or t in STOP_DEV:
            continue
        if len(t) >= (min_len_dev if script_of(t) == "dev" else min_len) and (script_of(t) == "dev" or t.isalpha()):
            out.append(t)
    return out


def grounding_overlap(hyp, doc_tokens, min_len: int = 4) -> float:
    """Share of the reply's content words (Latin or Devanagari, no stop words) that occur in the grounding document.
    Higher = the reply re-uses document wording. Returns nan if the reply has no content words."""
    doc = set(doc_tokens)
    content = content_words(hyp, min_len)
    if not content:
        return float("nan")
    return sum(t in doc for t in content) / len(content)


def evaluate_generation(refs, hyps, docs=None) -> dict:
    out = {**{k: v for k, v in corpus_bleu(refs, hyps).items() if k in ("bleu", "bp")}}
    for n in (1, 2):
        out[f"bleu{n}"] = corpus_bleu(refs, hyps, max_n=n)["bleu"]
    out.update(rouge_scores(refs, hyps))
    out["distinct1"], out["distinct2"] = distinct_n(hyps, 1), distinct_n(hyps, 2)
    out["avg_len"] = sum(len(h) for h in hyps) / max(len(hyps), 1)
    out["repetition"] = repetition_rate(hyps)
    if docs is not None:
        vals = [grounding_overlap(h, d) for h, d in zip(hyps, docs)]
        vals = [v for v in vals if v == v]
        out["grounding_overlap"] = sum(vals) / max(len(vals), 1)
    return out
