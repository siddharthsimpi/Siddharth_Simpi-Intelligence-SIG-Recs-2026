"""Data for sub-task 3: document-grounded Hinglish dialogue (CMU Hinglish DoG).

Canonical in-memory format (everything below works on this, whatever the raw source looked like):
    conversation = {"conv_id": str, "split": "train|val|test"|None, "doc_id": str,
                    "doc_sections": {"0": text, "1": text, ...}, "doc_full": text,
                    "turns": [{"speaker": 0|1, "hi_en": text, "en": text|None, "doc_idx": int|None}, ...]}
`load_conversations` adapts to the raw schema (HuggingFace rows, local JSON/JSONL/CSV/parquet, or our canonical JSONL).
If your copy of the data differs, run `python subtask3/inspect_dataset.py` and fix the few lines in `normalise_rows`.
"""
from __future__ import annotations

import glob
import json
import os
import urllib.request

import numpy as np

from .tokenize import Vocab, normalize, script_of, tokenize

HF_NAME = "festvox/cmu_hinglish_dog"
SPEAKER_TAGS = ["<usr1>", "<usr2>"]
SEP = "<sep>"
CONV_KEYS = ["conv_id", "conversation_id", "convId", "conversationId", "dialog_id", "dialogue_id", "chat_id", "id"]
DOC_TEXT_KEYS = ["document", "doc", "wiki_doc", "wikiDocument", "wiki_text", "docText", "doc_text", "passage", "knowledge"]
DOC_ID_KEYS = ["wikiDocumentIdx", "wiki_document_idx", "wikiDocumentId", "doc_id", "movie_id"]
SECTION_KEYS = ["docIdx", "doc_idx", "section", "section_idx"]


# --------------------------------------------------------------------------- raw loading
def _read_any(path):
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".json"):
        txt = open(path, encoding="utf-8").read()
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                obj = next((v for v in obj.values() if isinstance(v, list)), [obj])
            return obj
        except json.JSONDecodeError:
            return [json.loads(l) for l in txt.splitlines() if l.strip()]
    if ext in (".csv", ".tsv"):
        return pd.read_csv(path, sep="\t" if ext == ".tsv" else ",").to_dict("records")
    if ext == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    raise ValueError(path)


def _split_name(fn):
    n = os.path.basename(fn).lower()
    if "train" in n:
        return "train"
    if "valid" in n or "val" in n or "dev" in n:
        return "val"
    if "test" in n:
        return "test"
    return None


def load_local_rows(folder):
    rows = []
    for f in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True)):
        if f.lower().endswith((".jsonl", ".json", ".csv", ".tsv", ".parquet")) and "canonical" not in os.path.basename(f):
            try:
                part = _read_any(f)
            except Exception as e:  # noqa: BLE001
                print(f"[data] skipped {os.path.basename(f)}: {type(e).__name__}: {str(e)[:80]}")
                continue
            sp = _split_name(f)
            for r in part:
                if isinstance(r, dict):
                    r = dict(r)
                    if sp and "split" not in r:
                        r["__split__"] = sp
                    rows.append(r)
            print(f"[data] read {len(part):,} records from {os.path.basename(f)}")
    return rows


def download_hf_rows(data_dir):
    """Rows of the HF dataset: (1) `datasets` library, (2) parquet conversion API, (3) raw repo files."""
    rows = []
    try:
        from datasets import load_dataset
    except ImportError:
        load_dataset = None
        print("[data] the `datasets` package is not installed (pip install datasets); trying direct download ...")
    attempts = [] if load_dataset is None else [
        ("load_dataset", dict()), ("load_dataset(trust_remote_code)", dict(trust_remote_code=True)),
        ("load_dataset(parquet branch)", dict(revision="refs/convert/parquet"))]
    for label, kw in attempts:
        try:
            ds = load_dataset(HF_NAME, **kw)
            for split in ds:
                sp = {"validation": "val"}.get(split, split)
                for r in ds[split]:
                    r = dict(r); r["__split__"] = sp; rows.append(r)
            print(f"[data] loaded {len(rows):,} rows with {label}")
            return rows
        except Exception as e:  # noqa: BLE001
            rows = []
            print(f"[data] {label} failed: {type(e).__name__}: {str(e)[:220]}")
    print("[data] trying direct download ...")
    folder = os.path.join(data_dir, "hinglish_dog")
    os.makedirs(folder, exist_ok=True)

    def get(url):
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    try:
        info = json.loads(get(f"https://huggingface.co/api/datasets/{HF_NAME}/parquet"))
        for cfg, splits in info.items():
            for split, urls in splits.items():
                for i, u in enumerate(urls):
                    dst = os.path.join(folder, f"{split}_{i}.parquet")
                    if not os.path.exists(dst):
                        open(dst, "wb").write(get(u))
        return load_local_rows(folder)
    except Exception as e:  # noqa: BLE001
        print(f"[data] parquet route failed ({type(e).__name__}); trying the raw repo files ...")
    meta = json.loads(get(f"https://huggingface.co/api/datasets/{HF_NAME}"))
    for s in meta.get("siblings", []):
        fn = s["rfilename"]
        if fn.lower().endswith((".json", ".jsonl", ".csv", ".parquet", ".tsv")):
            dst = os.path.join(folder, fn.replace("/", "_"))
            if not os.path.exists(dst):
                open(dst, "wb").write(get(f"https://huggingface.co/datasets/{HF_NAME}/resolve/main/{fn}"))
    return load_local_rows(folder)


# --------------------------------------------------------------------------- wiki documents
def _doc_key(x):
    """Canonical form of a document id so that 4, 4.0, '4' and ' 4 ' all match."""
    s = str(x).strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return s.lower()


def _flat_text(v):
    """Recursively join every string inside a nested dict / list."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return " ".join(t for t in (_flat_text(x) for x in v.values()) if t)
    if isinstance(v, (list, tuple)):
        return " ".join(t for t in (_flat_text(x) for x in v) if t)
    return ""


def _norm_doc(obj):
    """One Wikipedia document -> {"sections": {"0": text, ...}, "full": text}. Digit keys are the sections the speakers were shown
    (docIdx 0..3 in CMU DoG); other keys (introduction, scene, cast, ...) are used for the full text when there are no digit keys."""
    if isinstance(obj, str):
        return {"sections": {"0": obj}, "full": obj}
    items = list(obj.items()) if isinstance(obj, dict) else list(enumerate(obj))
    sec, parts = {}, []
    for k, v in items:
        txt = _flat_text(v)
        if not txt:
            continue
        parts.append(txt)
        if str(k).isdigit():
            sec[str(k)] = txt
    full = " ".join(sec[k] for k in sorted(sec, key=int)) if sec else " ".join(parts)
    return {"sections": sec, "full": full}


def _wiki_folder(path):
    """If `path` is e.g. the root of the CMU DoG repo, descend into its WikiData folder."""
    if not os.path.isdir(path) or "wiki" in os.path.basename(os.path.normpath(path)).lower():
        return path
    for p in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)):
        if os.path.isdir(p) and "wiki" in os.path.basename(p).lower() and glob.glob(os.path.join(p, "**", "*.json"), recursive=True):
            print(f"[data] --docs: using the WikiData folder {p}")
            return p
    return path


def load_wiki_docs(path):
    """{doc_id: {"sections": {...}, "full": text}} from a JSON file (dict or list of documents) or a folder of <id>.json files.
    Besides the file name, a document is also registered under an id stored inside it (wikiDocumentIdx / doc_id / id)."""
    out = {}

    def add(key, doc, aliases=()):
        for k in (key,) + tuple(aliases):
            if k is not None and str(k).strip() != "":
                out.setdefault(_doc_key(k), doc)

    if os.path.isdir(path):
        path = _wiki_folder(path)
        files = sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
        if not files:
            found = sorted({os.path.splitext(f)[1] for f in glob.glob(os.path.join(path, "**", "*"), recursive=True) if os.path.isfile(f)})
            raise FileNotFoundError(f"no .json document files under {path} (file types found there: {found or 'none'}). "
                                    "--docs must point to the WikiData folder (or a JSON file) of the CMU DoG repository.")
        for f in files:
            try:
                obj = json.load(open(f, encoding="utf-8"))
            except (OSError, ValueError):
                continue
            stem = os.path.splitext(os.path.basename(f))[0]
            inner = [obj.get(k) for k in ("wikiDocumentIdx", "wiki_document_idx", "doc_id", "docId") if isinstance(obj, dict)]
            add(stem, _norm_doc(obj), [x for x in inner if isinstance(x, (str, int, float))])
    else:
        obj = json.load(open(path, encoding="utf-8"))
        if isinstance(obj, dict) and obj and not all(isinstance(v, (dict, list)) for v in obj.values()):
            add(os.path.splitext(os.path.basename(path))[0], _norm_doc(obj))          # the file is ONE document
        else:
            for k, v in (obj.items() if isinstance(obj, dict) else enumerate(obj)):
                add(k, _norm_doc(v))
    out = {k: v for k, v in out.items() if v["full"]}
    if not out:
        raise ValueError(f"found document files under {path} but none contained any text")
    n_unique = len({id(v) for v in out.values()})
    print(f"[data] loaded {n_unique} grounding documents; ids such as {sorted(out)[:6]}")
    return out


# --------------------------------------------------------------------------- schema adaptation
def _first(row, keys):
    for k in keys:
        if k in row and row[k] is not None and not (isinstance(row[k], float) and np.isnan(row[k])):
            return row[k]
    return None


NESTED_KEYS = ["history", "turns", "dialog", "dialogue", "conversation", "utterances", "utterance_list", "chat", "messages"]


def flatten_rows(rows):
    """Make every record a single turn. Handles records that are whole conversations: either a nested list of turn dicts
    (`history`, `turns`, ...) or a `translation` dict whose values are parallel lists (one entry per turn)."""
    out = []
    for n, r in enumerate(rows):
        tr = r.get("translation")
        nested = next((k for k in NESTED_KEYS if isinstance(r.get(k), list) and r[k] and isinstance(r[k][0], (dict, str))), None)
        base = {k: v for k, v in r.items() if not isinstance(v, (list, dict)) or k in ("translation",)}
        cid = next((r[k] for k in CONV_KEYS if k in r and k != "id"), f"row{n}")
        if isinstance(tr, dict) and tr and all(isinstance(v, list) for v in tr.values()):
            length = len(next(iter(tr.values())))
            lists = {k: v for k, v in r.items() if isinstance(v, list) and len(v) == length and k != "translation"}
            for i in range(length):
                t = dict(base)
                t["translation"] = {k: v[i] for k, v in tr.items()}
                t.update({k: v[i] for k, v in lists.items()})
                t["conversation_id"] = cid
                out.append(t)
        elif nested:
            for el in r[nested]:
                t = dict(base)
                t.update(el if isinstance(el, dict) else {"text": el})
                t["conversation_id"] = cid
                out.append(t)
        else:
            out.append(r)
    return out


def normalise_rows(rows, wiki_docs=None, seed=0):
    """Raw per-turn rows -> canonical conversations. Prints what it detected so mismatches are easy to spot."""
    if not rows:
        raise ValueError("no rows loaded")
    rows = flatten_rows(rows)
    wiki_lookup = {_doc_key(k): v for k, v in wiki_docs.items()} if wiki_docs else None
    r0 = rows[0]
    print("[data] first raw record keys:", list(r0.keys()))
    conv_key = next((k for k in CONV_KEYS if k in r0 and k != "id"), None) or ("id" if "id" in r0 and len(rows) > 1 and
                    len({r.get("id") for r in rows}) < len(rows) else None)
    convs, order = {}, []
    if conv_key is None and "uid1LogInTime" in r0 and "user2_id" in r0:
        # CMU DoG rows carry no conversation id, but every turn repeats its session's fields: they identify the conversation exactly
        for r in rows:
            r["__conv__"] = "|".join(str(r.get(k)) for k in ("__split__", "wikiDocumentIdx", "uid1LogInTime", "uid1LogOutTime", "user2_id"))
        conv_key = "__conv__"
        print("[data] no conversation-id field: grouping turns by session (wikiDocumentIdx, uid1LogInTime, uid1LogOutTime, user2_id)")
    if conv_key is None:
        print("[data] WARNING: no conversation-id field found; grouping consecutive turns by (document, timestamp gap).")
        cur, last_doc, last_t, last_split = 0, None, None, None
        for r in rows:
            doc = _first(r, DOC_ID_KEYS)
            ts = _first(r, ["utcTimestamp", "timestamp"])
            try:
                import pandas as pd
                t = pd.Timestamp(ts) if ts else None
            except Exception:  # noqa: BLE001
                t = None
            if last_doc is not None and (doc != last_doc or r.get("__split__") != last_split or (t is not None and last_t is not None and abs((t - last_t).total_seconds()) > 3600)):
                cur += 1
            r["__conv__"], last_doc, last_t, last_split = f"c{cur}", doc, t, r.get("__split__")
        conv_key = "__conv__"
    else:
        print(f"[data] conversation id field: '{conv_key}'")
    for r in rows:
        cid = str(r[conv_key])
        if cid not in convs:
            convs[cid] = []
            order.append(cid)
        convs[cid].append(r)
    out, missing_ids = [], set()
    for cid in order:
        rs = convs[cid]
        if any(_first(r, ["utcTimestamp"]) for r in rs):
            rs = sorted(rs, key=lambda r: str(_first(r, ["utcTimestamp"]) or ""))
        turns, spk_map = [], {}
        doc_id, sections, full = None, {}, ""
        for r in rs:
            tr = r.get("translation")
            if isinstance(tr, dict):
                hi_en, en = tr.get("hi_en") or tr.get("hinglish"), tr.get("en")
            else:
                hi_en, en = _first(r, ["hi_en", "hinglish", "text", "utterance"]), _first(r, ["en", "english"])
            if not hi_en or not str(hi_en).strip():
                continue
            spk = _first(r, ["uid", "speaker", "user", "user_id"])
            spk_map.setdefault(str(spk), len(spk_map) % 2)
            sec = _first(r, SECTION_KEYS)
            turns.append({"speaker": spk_map[str(spk)], "hi_en": str(hi_en), "en": None if en is None else str(en),
                          "doc_idx": None if sec is None else int(sec)})
            if doc_id is None:
                doc_id = _first(r, DOC_ID_KEYS)
            dt = _first(r, DOC_TEXT_KEYS)
            if dt is not None and not full:
                if isinstance(dt, dict):
                    sections = {str(k): str(v) for k, v in dt.items() if str(k).isdigit()}
                    full = " ".join(map(str, dt.values()))
                else:
                    full = str(dt)
        if wiki_lookup is not None and doc_id is not None and not full:
            hit = wiki_lookup.get(_doc_key(doc_id))
            if hit is not None:
                sections, full = hit["sections"], hit["full"]
            else:
                missing_ids.add(str(doc_id))
        if len(turns) >= 2:
            out.append({"conv_id": cid, "split": rs[0].get("__split__") or rs[0].get("split"), "doc_id": str(doc_id),
                        "doc_sections": sections, "doc_full": full, "turns": turns})
    n_doc = sum(1 for c in out if c["doc_full"])
    print(f"[data] {len(out):,} conversations, {sum(len(c['turns']) for c in out):,} turns, "
          f"{n_doc:,} conversations with grounding text")
    if wiki_lookup is not None and missing_ids:
        print(f"[data] WARNING: {len(missing_ids)} document id(s) used by the dialogues were not found among the documents: "
              f"{sorted(missing_ids)[:8]} ... documents are keyed by {sorted(wiki_lookup)[:8]} ...")
    if n_doc == 0:
        ids_in_rows = sorted({str(_first(r, DOC_ID_KEYS)) for r in rows[:5000]})[:8]
        raise RuntimeError(
            "The dataset rows contain no grounding-document text and none could be attached.\n"
            f"  document ids used by the dialogue rows (field wikiDocumentIdx): {ids_in_rows}\n"
            f"  document ids that were loaded from --docs: {sorted(wiki_lookup)[:8] if wiki_lookup else 'none (no --docs given)'}\n"
            "If the two lists look different, --docs points at the wrong folder or the files are named differently: pass the\n"
            "WikiData folder of https://github.com/festvox/datasets-CMU_DoG (or the repo root), or a JSON file {id: document}.\n"
            "Otherwise fix DOC_TEXT_KEYS / DOC_ID_KEYS in s2s/data_dialogue.py. Run subtask3/inspect_dataset.py to see the fields.")
    if n_doc < len(out):                                      # a grounded model needs a document for every example
        print(f"[data] dropping {len(out) - n_doc} conversation(s) without a grounding document")
        out = [c for c in out if c["doc_full"]]
    return out


CMU_DOG_ZIP = "https://github.com/festvox/datasets-CMU_DoG/archive/refs/heads/master.zip"


def _long_path(p):
    """Windows: lift the 260-character path limit with the extended-length prefix (no-op elsewhere)."""
    p = os.path.abspath(p)
    return "\\\\?\\" + p if os.name == "nt" and not p.startswith("\\\\?\\") else p


def fetch_cmu_dog_wikidata(data_dir):
    """Download the original CMU DoG repository and extract ONLY its WikiData documents, flat, into a short folder.
    (Extracting the whole repository fails on Windows when the project sits in a deep folder such as OneDrive\\Desktop\\...,
    because the repo's Conversations files have long names and the full path exceeds 260 characters.)"""
    import io
    import tempfile
    import zipfile
    dst = os.path.join(data_dir, "cmu_dog_wikidata")
    if not glob.glob(os.path.join(dst, "*.json")):
        print(f"[data] fetching the grounding documents from {CMU_DOG_ZIP} ...")
        with urllib.request.urlopen(CMU_DOG_ZIP, timeout=120) as r:
            payload = r.read()
        zf = zipfile.ZipFile(io.BytesIO(payload))
        members = [m for m in zf.namelist()
                   if m.lower().endswith(".json") and any("wiki" in part.lower() for part in m.split("/")[:-1])]
        if not members:
            raise FileNotFoundError("the downloaded repository archive contains no WikiData folder")
        last_err = None
        for target in (dst, os.path.join(tempfile.gettempdir(), "cmu_dog_wikidata")):
            try:
                os.makedirs(_long_path(target), exist_ok=True)
                for m in members:
                    with open(_long_path(os.path.join(target, os.path.basename(m))), "wb") as f:
                        f.write(zf.read(m))
                dst = target
                break
            except OSError as e:
                last_err = e
                print(f"[data] could not write to {target} ({type(e).__name__}: {e}); trying another folder ...")
        else:
            raise last_err
    print(f"[data] using grounding documents from {dst}")
    return load_wiki_docs(dst)


def verify_doc_alignment(convs, wiki_docs, min_ratio=1.25):
    """Sanity check: is the document attached to each conversation really the one the conversation is about? For each document id,
    the conversation text is compared (idf-weighted word overlap) with every loaded document. If the attached document is not
    the best match for most ids but a clear, consistent one-to-one re-mapping exists (e.g. ids shifted by one), it is applied -
    otherwise a warning is printed. Wrong grounding documents would silently invalidate every grounded result."""
    import math
    from collections import Counter
    from .metrics import content_words
    uniq = {}
    for k, v in wiki_docs.items():
        uniq.setdefault(id(v), (k, v))
    docs = list(uniq.values())
    if len(docs) < 3 or not convs:
        return convs
    dsets = [set(content_words(tokenize(v["full"]))) for _, v in docs]
    df = Counter(w for st_ in dsets for w in st_)
    idf = {w: math.log(len(docs) / c) for w, c in df.items()}
    by_id = {}
    for c in convs:
        by_id.setdefault(c["doc_id"], []).append(c)
    best, assigned_ok, confident = {}, 0, 0
    for did, cs in by_id.items():
        words = set(content_words(tokenize(" ".join((t.get("en") or t["hi_en"]) for c in cs for t in c["turns"]))))
        sc = sorted(((sum(idf.get(w, 0.0) for w in words & ds) / math.sqrt(len(ds) + 1), i) for i, ds in enumerate(dsets)), reverse=True)
        if len(sc) > 1 and sc[0][0] > 0 and sc[0][0] >= min_ratio * max(sc[1][0], 1e-9):
            confident += 1
            best[did] = sc[0][1]
            assigned_ok += docs[sc[0][1]][1]["full"] == cs[0]["doc_full"]
    if confident == 0:
        print("[data] document-alignment check: inconclusive (no clear match between conversations and documents)")
        return convs
    print(f"[data] document-alignment check: for {assigned_ok}/{confident} confidently matched document ids the attached document is the best match")
    if assigned_ok >= 0.7 * confident:
        return convs
    if confident >= 0.8 * len(by_id) and len(set(best.values())) == len(best):
        print("[data] WARNING: the document ids do not line up with the document files; re-mapping by content:")
        for did, i in sorted(best.items(), key=lambda kv: str(kv[0])):
            print(f"         dialogue document id {did} -> document file '{docs[i][0]}'")
        for did, i in best.items():
            for c in by_id[did]:
                c["doc_sections"], c["doc_full"] = docs[i][1]["sections"], docs[i][1]["full"]
        return convs
    print("[data] WARNING: many conversations seem to be paired with the WRONG document and no consistent re-mapping was found. "
          "Check that --docs points at the CMU DoG WikiData folder and that its file names are the document ids.")
    return convs


def assign_splits(convs, seed=0, fractions=(0.8, 0.1, 0.1)):
    if all(c["split"] in ("train", "val", "test") for c in convs):
        return convs
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(convs))
    n_val, n_test = int(len(convs) * fractions[1]), int(len(convs) * fractions[2])
    for rank, i in enumerate(order):
        convs[i]["split"] = "test" if rank < n_test else "val" if rank < n_test + n_val else "train"
    print("[data] no predefined splits found: split conversations randomly 80/10/10")
    return convs


def load_conversations(path=None, docs_path=None, data_dir="data", seed=0):
    canon = os.path.join(path or "", "canonical.jsonl")
    if path and os.path.isfile(canon):
        convs = [json.loads(l) for l in open(canon, encoding="utf-8") if l.strip()]
        return assign_splits(convs, seed), f"canonical file {canon}"
    wiki = load_wiki_docs(docs_path) if docs_path else None
    rows = load_local_rows(path) if path else download_hf_rows(data_dir)
    try:
        convs = normalise_rows(rows, wiki, seed)
    except RuntimeError as first_error:                       # rows carry no document text and --docs was not given
        if docs_path:
            raise
        print("[data] rows contain no document text; trying to fetch the original CMU DoG Wikipedia documents ...")
        try:
            wiki = fetch_cmu_dog_wikidata(data_dir)
            convs = normalise_rows(rows, wiki, seed)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{first_error}\n(automatic fetch failed: {type(e).__name__}: {e})") from e
    if wiki:
        convs = verify_doc_alignment(convs, wiki)
    return assign_splits(convs, seed), (f"local data {path}" if path else f"HuggingFace {HF_NAME}")


# --------------------------------------------------------------------------- code-mixing statistics
def english_share(turn) -> float | None:
    """Share of a Hinglish turn's words that are English - a proxy for how English-heavy the turn is. Latin-script words that
    also occur in the turn's own English translation count as English (English words / names); other Latin words are romanised
    Hindi; Devanagari words are Hindi and count in the denominator, so a Devanagari-only turn has share 0."""
    if not turn.get("en"):
        return None
    toks = [t for t in tokenize(turn["hi_en"])
            if (script_of(t) == "dev") or (script_of(t) == "lat" and t.isalpha() and len(t) > 1)]
    if not toks:
        return None
    en = set(tokenize(turn["en"]))
    return sum(t in en for t in toks if script_of(t) == "lat") / len(toks)


def mix_bucket(share):
    if share is None:
        return "unknown"
    return "mostly English" if share >= 0.6 else "mixed" if share >= 0.3 else "mostly Hindi"


# --------------------------------------------------------------------------- example construction
def conv_stats(convs):
    out = {"conversations": len(convs), "turns": sum(len(c["turns"]) for c in convs)}
    toks = [len(tokenize(t["hi_en"])) for c in convs for t in c["turns"]]
    out["avg_turn_tokens"], out["p95_turn_tokens"] = float(np.mean(toks)), float(np.percentile(toks, 95))
    sh = [english_share(t) for c in convs for t in c["turns"]]
    sh = [s for s in sh if s is not None]
    out["english_share_mean"] = float(np.mean(sh)) if sh else None
    out["buckets"] = {b: sum(mix_bucket(s) == b for s in sh) for b in ("mostly English", "mixed", "mostly Hindi")}
    out["doc_tokens_mean"] = float(np.mean([len(tokenize(c["doc_full"])) for c in convs if c["doc_full"]] or [0]))
    dev = sum(script_of(t) == "dev" for c in convs for turn in c["turns"] for t in tokenize(turn["hi_en"]))
    out["devanagari_token_share"] = dev / max(sum(toks), 1)
    return out


def _doc_tokens(conv, doc_idx, max_doc_len):
    sec = conv["doc_sections"].get(str(doc_idx)) if doc_idx is not None else None
    toks = tokenize(sec if sec else conv["doc_full"])
    return toks[:max_doc_len]


def make_examples(convs, hist_turns=3, max_hist_len=60, max_resp_len=30, max_doc_len=100, field="hi_en"):
    """One example per response turn t >= 1: history = previous <= hist_turns turns (speaker-tagged), doc = the section the
    responder was reading (falls back to the document start)."""
    ex = []
    for c in convs:
        for t in range(1, len(c["turns"])):
            if not c["turns"][t].get(field):                  # field="en": turns without an English version are skipped
                continue
            resp = tokenize(c["turns"][t][field])
            if not 1 <= len(resp) <= max_resp_len:
                continue
            hist = []
            for j in range(max(0, t - hist_turns), t):
                hist += [SPEAKER_TAGS[c["turns"][j]["speaker"]]] + tokenize(c["turns"][j].get(field) or c["turns"][j]["hi_en"])
            hist = hist[-max_hist_len:]
            ex.append({"conv": c["conv_id"], "hist_tok": hist, "doc_tok": _doc_tokens(c, c["turns"][t]["doc_idx"], max_doc_len),
                       "resp_tok": resp, "share": english_share(c["turns"][t]) if field == "hi_en" else None, "turn": t, "lang": field})
    return ex


def build_dialogue_data(convs, max_train=None, max_eval=None, seed=0, min_freq=2, max_vocab=8000, add_english=False, **kw):
    split = {k: [c for c in convs if c["split"] == k] for k in ("train", "val", "test")}
    ex = {k: make_examples(v, **kw) for k, v in split.items()}
    if add_english:        # every turn also exists in English: use it as EXTRA TRAINING data only (val/test stay Hinglish)
        ex["train"] += make_examples(split["train"], field="en", **kw)
    rng = np.random.default_rng(seed)
    if max_train and len(ex["train"]) > max_train:
        ex["train"] = [ex["train"][i] for i in rng.permutation(len(ex["train"]))[:max_train]]
    for k in ("val", "test"):
        if max_eval and len(ex[k]) > max_eval:
            ex[k] = [ex[k][i] for i in rng.permutation(len(ex[k]))[:max_eval]]
    # vocabulary from TRAIN conversations only (turns + documents) so that test words are genuinely unseen
    texts = []
    for c in split["train"]:
        texts += [tokenize(t["hi_en"]) for t in c["turns"]]
        if add_english:
            texts += [tokenize(t["en"]) for t in c["turns"] if t.get("en")]
        texts += [tokenize(c["doc_full"])]
        texts += [tokenize(s) for s in c["doc_sections"].values()]
    vocab = Vocab.build(texts + [e["hist_tok"] for e in ex["train"]], min_freq, max_vocab, extra_specials=SPEAKER_TAGS + [SEP])
    for k, lst in ex.items():
        for e in lst:
            e["srcs"] = [vocab.encode(e["hist_tok"]), vocab.encode(e["doc_tok"]) or [vocab.unk]]
            e["tgt"] = vocab.encode(e["resp_tok"])
    return {"vocab": vocab, "examples": ex, "train_texts": texts,
            "stats": {k: len(v) for k, v in ex.items()} | {"vocab": len(vocab),
                      "resp_oov_test": vocab.oov_rate([e["resp_tok"] for e in ex["test"]])}}


# --------------------------------------------------------------------------- toy corpus (tests / smoke runs only)
def make_toy_dog(n_conv=400, seed=0):
    """SYNTHETIC grounded Hinglish-style conversations about invented movies - for tests and smoke runs only.
    Facts asked about in the conversation (director / year / actor / genre) can only be answered from the document."""
    rng = np.random.default_rng(seed)
    titles = ["blue horizon", "silent river", "red desert", "iron city", "golden road", "night train", "paper moon",
              "last summer", "broken wings", "hidden door", "storm front", "winter song"]
    dirs = ["rao", "kapoor", "mehta", "nair", "singh", "iyer", "khan", "das"]
    acts = ["arjun", "meera", "vikram", "anita", "rohan", "sneha", "karan", "pooja"]
    genres = ["drama", "thriller", "comedy", "romance", "action"]
    docs = {}
    for i, t in enumerate(titles):
        d, y, a, g = dirs[i % 8], 1990 + int(rng.integers(0, 30)), acts[(i * 3) % 8], genres[i % 5]
        docs[i] = {"title": t, "dir": d, "year": y, "act": a, "genre": g}
    convs = []
    for k in range(n_conv):
        did = int(rng.integers(0, len(titles)))
        m = docs[did]
        sections = {"0": f"{m['title']} is a {m['genre']} film directed by {m['dir']} and released in {m['year']} .",
                    "1": f"the film stars {m['act']} in the lead role . critics praised the story and the music ."}
        full = sections["0"] + " " + sections["1"]
        qa = [
            (f"kya tumne {m['title']} dekhi hai ?", f"have you seen {m['title']} ?", f"haan , {m['title']} mujhe pasand hai .", f"yes , i like {m['title']} .", 0),
            ("iska director kaun hai ?", "who is its director ?", f"director {m['dir']} hai . bahut accha kaam kiya .", f"the director is {m['dir']} . did a great job .", 0),
            ("yeh kab release hui thi ?", "when was it released ?", f"yeh {m['year']} mein release hui thi .", f"it was released in {m['year']} .", 0),
            ("isme lead role kisne kiya ?", "who played the lead ?", f"lead role mein {m['act']} hai . acting kamaal ki hai .", f"{m['act']} plays the lead . the acting is great .", 1),
            ("yeh kis genre ki movie hai ?", "what genre is it ?", f"yeh ek {m['genre']} movie hai , maza aayega .", f"it is a {m['genre']} movie , you will enjoy it .", 0),
            ("critics ko kaisi lagi ?", "how did critics like it ?", "critics ne story aur music ki tareef ki .", "critics praised the story and the music .", 1),
        ]
        order = rng.permutation(len(qa))[: int(rng.integers(3, 6))]
        turns = []
        for j in order:
            q_hi, q_en, a_hi, a_en, sec = qa[j]
            turns.append({"speaker": 0, "hi_en": q_hi, "en": q_en, "doc_idx": sec})
            turns.append({"speaker": 1, "hi_en": a_hi, "en": a_en, "doc_idx": sec})
        convs.append({"conv_id": f"toy{k}", "split": None, "doc_id": str(did), "doc_sections": sections,
                      "doc_full": full, "turns": turns})
    return assign_splits(convs, seed)
