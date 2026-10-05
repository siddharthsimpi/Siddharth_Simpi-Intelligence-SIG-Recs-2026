"""Parallel-corpus loading for sub-tasks 1-2 (English -> Hindi by default; any language pair works).

The loader is deliberately format-tolerant because the course data is a shared folder of unknown layout:
  * CSV / TSV with a source column and a target column (auto-detected from names such as english/hindi/en/hi)
  * Anki / Tatoeba style TXT:   source <TAB> target [<TAB> attribution]
  * JSON / JSONL records:       {"translation": {"en": ..., "hi": ...}}  or  {"en": ..., "hi": ...}
  * line-aligned file pairs:    train.en + train.hi
Files whose names contain train / val|valid|dev / test are used as the corresponding split; otherwise everything is pooled
and split 90 / 5 / 5 at random (seeded).
"""
from __future__ import annotations

import glob
import io
import json
import os
import urllib.request
import zipfile

import numpy as np

from .tokenize import Vocab, tokenize

SRC_ALIASES = {"en": ["en", "eng", "english", "source", "src", "input"], "hi": ["hi", "hin", "hindi"],
               "fr": ["fr", "fra", "french", "francais", "fran\u00e7ais"], "de": ["de", "deu", "german", "deutsch"],
               "es": ["es", "spa", "spanish", "espanol"], "it": ["it", "ita", "italian"], "pt": ["pt", "por", "portuguese"],
               "ru": ["ru", "rus", "russian"], "ta": ["ta", "tam", "tamil"], "bn": ["bn", "ben", "bengali"],
               "te": ["te", "tel", "telugu"], "mr": ["mr", "mar", "marathi"], "ur": ["ur", "urd", "urdu"]}
LANG_NAMES = {"en": "English", "hi": "Hindi", "fr": "French", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
              "ru": "Russian", "ta": "Tamil", "bn": "Bengali", "te": "Telugu", "mr": "Marathi", "ur": "Urdu"}


def lang_name(code):
    return LANG_NAMES.get(code, code)


def infer_langs(path, default=("en", "hi")):
    """Guess (source, target) language codes from file names such as en_fr_train.csv / train.en-hi.txt."""
    import re
    names = [os.path.basename(f) for f in (glob.glob(os.path.join(path, "**", "*"), recursive=True) if os.path.isdir(path) else [path])]
    for n in names:
        m = re.search(r"(?:^|[^a-z])([a-z]{2})[_\-]([a-z]{2})(?:[^a-z]|$)", n.lower())
        if m and m.group(1) in LANG_NAMES and m.group(2) in LANG_NAMES:
            return m.group(1), m.group(2)
    return default
SPLIT_WORDS = {"train": ["train"], "val": ["val", "valid", "dev"], "test": ["test"]}
DATA_EXT = (".csv", ".tsv", ".txt", ".json", ".jsonl", ".parquet")
ANKI_URL = "https://www.manythings.org/anki/hin-eng.zip"


def _aliases(lang):
    return SRC_ALIASES.get(lang, [lang])


def _pick_columns(cols, src_lang, tgt_lang, src_col, tgt_col):
    low = {c: str(c).strip().lower() for c in cols}
    def find(aliases):
        for a in aliases:                              # exact name first, then substring
            for c, l in low.items():
                if l == a:
                    return c
        for a in aliases:
            for c, l in low.items():
                if a in l and len(a) > 2:
                    return c
        return None
    s = src_col or find(_aliases(src_lang))
    t = tgt_col or find(_aliases(tgt_lang))
    if (s is None or t is None or s == t) and len(cols) == 2:
        s, t = cols[0], cols[1]
    if s is None or t is None or s == t:
        raise ValueError(f"Cannot tell which columns hold {src_lang}/{tgt_lang} text; columns are {list(cols)}. "
                         f"Pass --src-col / --tgt-col.")
    return s, t


def _read_file(path, src_lang, tgt_lang, src_col, tgt_col):
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", encoding="utf-8", on_bad_lines="skip",
                         dtype=str, keep_default_na=False)
        try:
            s, t = _pick_columns(list(df.columns), src_lang, tgt_lang, src_col, tgt_col)
        except ValueError:                                   # fall back to the first two text columns
            text_cols = [c for c in df.columns if str(c).lower() not in ("id", "index") and not str(c).lower().startswith("unnamed")]
            if len(text_cols) < 2:
                raise
            s, t = text_cols[0], text_cols[1]
        print(f"[data] {os.path.basename(path)}: using column '{s}' as {src_lang} and '{t}' as {tgt_lang}")
        return list(zip(df[s], df[t]))
    if ext == ".txt":
        pairs = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    pairs.append((parts[0], parts[1]))
        return pairs
    if ext in (".json", ".jsonl"):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        try:
            recs = json.loads(text)
            recs = recs if isinstance(recs, list) else next(v for v in recs.values() if isinstance(v, list))
        except json.JSONDecodeError:
            recs = [json.loads(l) for l in text.splitlines() if l.strip()]
        pairs = []
        for r in recs:
            r = r.get("translation", r)
            s, t = _pick_columns(list(r.keys()), src_lang, tgt_lang, src_col, tgt_col)
            pairs.append((str(r[s]), str(r[t])))
        return pairs
    if ext == ".parquet":
        df = pd.read_parquet(path)
        s, t = _pick_columns(list(df.columns), src_lang, tgt_lang, src_col, tgt_col)
        return list(zip(df[s].astype(str), df[t].astype(str)))
    raise ValueError(f"unsupported file type {path}")


def _split_of(name):
    n = os.path.basename(name).lower()
    for split, words in SPLIT_WORDS.items():
        if any(w in n for w in words):
            return split
    return "all"


def load_parallel(path, src_lang="en", tgt_lang="hi", src_col=None, tgt_col=None):  # noqa: C901
    """Returns {split: [(src_text, tgt_text), ...]} with splits among train/val/test/all."""
    files = [path] if os.path.isfile(path) else sorted(
        f for f in glob.glob(os.path.join(path, "**", "*"), recursive=True) if os.path.isfile(f))
    out: dict[str, list] = {}
    used = []
    # line-aligned file pairs  X.<src_lang> / X.<tgt_lang>
    by_stem = {}
    for f in files:
        stem, ext = os.path.splitext(f)
        if ext.lower().lstrip(".") in _aliases(src_lang) + _aliases(tgt_lang):
            by_stem.setdefault(stem, {})[ext.lower().lstrip(".")] = f
    for stem, d in by_stem.items():
        s = next((d[a] for a in _aliases(src_lang) if a in d), None)
        t = next((d[a] for a in _aliases(tgt_lang) if a in d), None)
        if s and t:
            with open(s, encoding="utf-8", errors="replace") as fs, open(t, encoding="utf-8", errors="replace") as ft:
                pairs = [(a.strip(), b.strip()) for a, b in zip(fs, ft) if a.strip() and b.strip()]
            out.setdefault(_split_of(stem), []).extend(pairs); used.append(stem)
    for f in files:
        if f.lower().endswith(DATA_EXT) and not any(f.startswith(u) for u in used):
            try:
                pairs = _read_file(f, src_lang, tgt_lang, src_col, tgt_col)
            except Exception as e:  # noqa: BLE001
                print(f"[data] skipped {os.path.basename(f)}: {type(e).__name__}: {str(e)[:100]}")
                continue
            if len(pairs) >= 2:
                out.setdefault(_split_of(f), []).extend(pairs); used.append(f)
                print(f"[data] read {len(pairs):,} pairs from {os.path.relpath(f, path) if os.path.isdir(path) else f}")
    if not out:
        raise FileNotFoundError(f"No parallel text found under {path}")
    return out


def download_anki(data_dir):
    """Tatoeba/Anki English-Hindi pairs (a few thousand short sentences) as a fallback corpus."""
    os.makedirs(data_dir, exist_ok=True)
    with urllib.request.urlopen(ANKI_URL, timeout=60) as r:
        payload = r.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(data_dir)
    return data_dir


def try_gdown(url, out_dir):
    try:
        import gdown
    except ImportError:
        return False
    try:
        gdown.download_folder(url, output=out_dir, quiet=False, use_cookies=False)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[data] gdown failed: {type(e).__name__}: {e}")
        return False


DRIVE_URL = "https://drive.google.com/drive/folders/1R4pOtvDEYyXtjLgVsT2RsmaORKOzoEq2"


def ensure_translation_data(path, data_dir, src_lang="en", tgt_lang="hi", src_col=None, tgt_col=None, allow_download=True):
    """Resolve the corpus: explicit path > data/translation/ > Google Drive folder (gdown) > Anki Hindi-English."""
    if path:
        return load_parallel(path, src_lang, tgt_lang, src_col, tgt_col), f"user data: {path}"
    folder = os.path.join(data_dir, "translation")
    have = os.path.isdir(folder) and any(os.path.isfile(f) for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True))
    if not have and allow_download:
        os.makedirs(folder, exist_ok=True)
        print("[data] data/translation/ is empty - trying the course Google Drive folder with gdown ...")
        if not try_gdown(DRIVE_URL, folder):
            print("[data] gdown unavailable/failed - downloading the Tatoeba/Anki English-Hindi pairs instead ...")
            try:
                download_anki(folder)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"Could not obtain a corpus automatically ({type(e).__name__}: {e}).\n"
                    f"Download the Drive folder manually in a browser ({DRIVE_URL}) and put its files into\n"
                    f"  {folder}\nthen re-run (or pass --data <folder>).") from e
    return load_parallel(folder, src_lang, tgt_lang, src_col, tgt_col), f"data folder: {folder}"


# --------------------------------------------------------------------------- toy corpus (tests / smoke runs only)
def make_toy_en_hi(n=3000, seed=0):
    """Tiny synthetic English -> pseudo-Hindi (Devanagari, SOV order). ONLY for tests and smoke runs - not a real corpus."""
    rng = np.random.default_rng(seed)
    subj = [("i", "मैं"), ("you", "तुम"), ("he", "वह"), ("she", "वह"), ("we", "हम"), ("they", "वे"),
            ("the boy", "लड़का"), ("the girl", "लड़की"), ("the teacher", "शिक्षक"), ("my mother", "माँ")]
    obj = [("apples", "सेब"), ("tea", "चाय"), ("books", "किताबें"), ("letters", "पत्र"), ("movies", "फिल्में"),
           ("songs", "गाने"), ("rice", "चावल"), ("water", "पानी")]
    verb = [("eat", "खाते"), ("drink", "पीते"), ("read", "पढ़ते"), ("write", "लिखते"), ("watch", "देखते"), ("like", "पसंद करते")]
    adv = [("", ""), ("quickly", "जल्दी"), ("slowly", "धीरे"), ("every day", "रोज़"), ("today", "आज")]
    pairs = []
    for _ in range(n):
        (se, sh), (oe, oh), (ve, vh), (ae, ah) = (subj[rng.integers(len(subj))], obj[rng.integers(len(obj))],
                                                  verb[rng.integers(len(verb))], adv[rng.integers(len(adv))])
        en = f"{se} {ve}{'s' if se not in ('i', 'you', 'we', 'they') else ''} {oe} {ae}".split()
        en = " ".join(en)
        hi = " ".join(w for w in [ah, sh, oh, vh, "हैं"] if w)
        pairs.append((en, hi))
    return {"all": pairs}


# --------------------------------------------------------------------------- dataset building
def build_translation_data(pairs_by_split, max_len=30, min_freq=2, max_vocab=8000, seed=0, max_train=None,
                           val_frac=0.05, test_frac=0.05, max_eval=None):
    def prep(pairs):
        out, seen = [], set()
        for s, t in pairs:
            st, tt = tokenize(s), tokenize(t)
            if not (1 <= len(st) <= max_len and 1 <= len(tt) <= max_len):
                continue
            key = (tuple(st), tuple(tt))
            if key in seen:
                continue
            seen.add(key)
            out.append((st, tt))
        return out

    rng = np.random.default_rng(seed)
    if max_train:                                   # huge corpora: tokenise only a random sample (10x the training cap)
        cap = max_train * 10
        pairs_by_split = {k: ([v[i] for i in rng.permutation(len(v))[:cap]] if len(v) > cap else v)
                          for k, v in pairs_by_split.items()}
    splits = {k: prep(v) for k, v in pairs_by_split.items()}
    if "train" not in splits:                                  # pool everything and split randomly
        pool = [p for v in splits.values() for p in v]
        order = rng.permutation(len(pool))
        n_val, n_test = int(len(pool) * val_frac), int(len(pool) * test_frac)
        pick = lambda a, b: [pool[i] for i in order[a:b]]
        splits = {"test": pick(0, n_test), "val": pick(n_test, n_test + n_val), "train": pick(n_test + n_val, len(pool))}
    else:
        for k in ("val", "test"):
            if k not in splits:                                # carve from train when missing
                tr = splits["train"]
                order = rng.permutation(len(tr))
                n = max(1, int(len(tr) * (val_frac if k == "val" else test_frac)))
                splits[k] = [tr[i] for i in order[:n]]
                splits["train"] = [tr[i] for i in order[n:]]
                tr = splits["train"]
    # leakage guard: drop validation/test pairs whose sentence pair also occurs in the training data
    train_keys = {(tuple(s), tuple(t)) for s, t in splits["train"]}
    removed = 0
    for k in ("val", "test"):
        kept = [(s, t) for s, t in splits[k] if (tuple(s), tuple(t)) not in train_keys]
        removed += len(splits[k]) - len(kept)
        splits[k] = kept
    if removed:
        print(f"[data] removed {removed} validation/test pairs that also occur in the training data")
    for k in ("val", "test"):                              # nothing usable left (e.g. a tiny, highly repetitive corpus)
        if 0 < len(splits[k]) < 50:
            print(f"[data] WARNING: the {k} split has only {len(splits[k])} pairs - scores will be noisy")
        if len(splits[k]) == 0:
            tr = splits["train"]
            order = rng.permutation(len(tr))
            n = max(1, int(len(tr) * (val_frac if k == "val" else test_frac)))
            splits[k] = [tr[i] for i in order[:n]]
            splits["train"] = [tr[i] for i in order[n:]]
            print(f"[data] WARNING: {k} split was empty after de-duplication; carved {n} pairs out of train instead")
    if max_train:
        splits["train"] = [splits["train"][i] for i in rng.permutation(len(splits["train"]))[:max_train]]
    if max_eval:
        for k in ("val", "test"):
            splits[k] = splits[k][:max_eval]
    sv = Vocab.build([s for s, _ in splits["train"]], min_freq, max_vocab)
    tv = Vocab.build([t for _, t in splits["train"]], min_freq, max_vocab)
    data = {"src_vocab": sv, "tgt_vocab": tv, "splits": {}}
    for k, pairs in splits.items():
        data["splits"][k] = [{"srcs": [sv.encode(s)], "tgt": tv.encode(t), "src_tok": s, "tgt_tok": t} for s, t in pairs]
    data["stats"] = {
        "n_train": len(splits["train"]), "n_val": len(splits["val"]), "n_test": len(splits["test"]),
        "src_vocab": len(sv), "tgt_vocab": len(tv),
        "avg_src_len": float(np.mean([len(s) for s, _ in splits["train"]])),
        "avg_tgt_len": float(np.mean([len(t) for _, t in splits["train"]])),
        "src_oov_test": sv.oov_rate([s for s, _ in splits["test"]]),
        "tgt_oov_test": tv.oov_rate([t for _, t in splits["test"]]),
    }
    return data
