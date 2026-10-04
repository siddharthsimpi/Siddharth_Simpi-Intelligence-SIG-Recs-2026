#!/usr/bin/env python
"""Print what the dialogue dataset looks like (fields, examples, document availability) BEFORE training.

    python inspect_dataset.py                     # downloads / loads festvox/cmu_hinglish_dog
    python inspect_dataset.py --data <folder> [--docs <wiki json or folder>]
If the loader complains, send this script's output to whoever maintains the code - it shows exactly which fields exist.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from s2s import data_dialogue as D  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--docs", default=None)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--toy", action="store_true")
    a = ap.parse_args()
    if a.toy:
        convs, src = D.make_toy_dog(50), "toy"
    else:
        rows = D.load_local_rows(a.data) if a.data else D.download_hf_rows(a.data_dir)
        print(f"\n{len(rows):,} raw records. First 2 records:")
        for r in rows[:2]:
            for k, v in r.items():
                print(f"   {k!r}: {str(v)[:110]!r}")
            print("   ---")
        ids = sorted({str(D._first(r, D.DOC_ID_KEYS)) for r in rows}, key=lambda x: (len(x), x))
        print(f"\ndocument ids used by the dialogue rows ({len(ids)} distinct): {ids[:12]}")
        has_text = [k for k in D.DOC_TEXT_KEYS if rows and k in rows[0]]
        print(f"fields in the rows that hold document text: {has_text or 'NONE (the documents must come from --docs or the CMU DoG repo)'}")
        if a.docs:
            try:
                wiki = D.load_wiki_docs(a.docs)
                print(f"documents loaded from --docs: {len(wiki)} keys, e.g. {sorted(wiki)[:12]}")
                missing = [i for i in ids if D._doc_key(i) not in wiki]
                print("every dialogue document id has a matching document" if not missing else f"NO matching document for ids: {missing[:12]}")
            except Exception as e:  # noqa: BLE001
                print(f"could not load --docs: {type(e).__name__}: {e}")
        try:
            convs, src = D.load_conversations(a.data, a.docs, a.data_dir)
        except Exception as e:  # noqa: BLE001
            print(f"\nLOADING FAILED: {type(e).__name__}: {e}")
            print("Copy everything printed above and send it to whoever maintains the code.")
            sys.exit(1)
    print(f"\nsource: {src}")
    st = D.conv_stats(convs)
    for k, v in st.items():
        print(f"  {k}: {v}")
    c = convs[0]
    print(f"\nexample conversation {c['conv_id']} (doc {c['doc_id']}, split {c['split']}):")
    print("  document start:", c["doc_full"][:200].replace("\n", " "))
    for t in c["turns"][:6]:
        print(f"  [{t['speaker']}] hi_en: {t['hi_en'][:90]}")
        print(f"      en   : {str(t['en'])[:90]}")


if __name__ == "__main__":
    main()
