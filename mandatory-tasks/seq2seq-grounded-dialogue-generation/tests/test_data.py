import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from s2s import data_dialogue as D  # noqa: E402
from s2s import data_translation as T  # noqa: E402
from s2s.baselines import TfidfIndex, extractive_baseline, retrieval_baseline  # noqa: E402
from s2s.decoding import _banned  # noqa: E402

EN = ["i like tea", "she reads books", "we watch movies", "he drinks water", "they eat rice", "you write letters"] * 6
HI = ["मुझे चाय पसंद है", "वह किताबें पढ़ती है", "हम फिल्में देखते हैं", "वह पानी पीता है", "वे चावल खाते हैं", "तुम पत्र लिखते हो"] * 6


def _tmp():
    return tempfile.mkdtemp()


def test_translation_loader_csv_with_named_columns():
    d = _tmp()
    import pandas as pd
    pd.DataFrame({"english_sentence": EN, "hindi_sentence": HI, "id": range(len(EN))}).to_csv(os.path.join(d, "all.csv"), index=False)
    out = T.load_parallel(d, "en", "hi")
    assert len(out["all"]) == len(EN) and out["all"][0] == (EN[0], HI[0])


def test_en_fr_csv_files_with_split_names_and_language_inference():
    import pandas as pd
    d = _tmp()
    base_en = ["the cat sleeps", "i love my family", "where is the station", "it is a beautiful day"]
    base_fr = ["le chat dort", "j'aime ma famille", "o\u00f9 est la gare", "c'est une belle journ\u00e9e"]
    def rows(offset, n):
        return ([f"{base_en[i % 4]} {offset + i}" for i in range(n)], [f"{base_fr[i % 4]} {offset + i}" for i in range(n)])
    for split, (off, n) in {"train": (0, 80), "val": (1000, 20), "test": (2000, 20)}.items():
        en, fr = rows(off, n)
        pd.DataFrame({"en": en, "fr": fr}).to_csv(os.path.join(d, f"en_fr_{split}.csv"), index=False)
    assert T.infer_langs(d) == ("en", "fr")
    out = T.load_parallel(d, "en", "fr")
    assert set(out) == {"train", "val", "test"} and out["train"][1] == (rows(0, 80)[0][1], rows(0, 80)[1][1])
    en, fr = rows(0, 40)
    d2 = _tmp()                                                     # Kaggle-style long column names
    pd.DataFrame({"English words/sentences": en, "French words/sentences": fr}).to_csv(os.path.join(d2, "x.csv"), index=False)
    assert T.load_parallel(d2, "en", "fr")["all"][2] == (en[2], fr[2])
    d3 = _tmp()                                                     # unknown column names: first two text columns
    pd.DataFrame({"id": range(40), "a": en, "b": fr}).to_csv(os.path.join(d3, "x.csv"), index=False)
    assert T.load_parallel(d3, "en", "fr")["all"][3] == (en[3], fr[3])
    data = T.build_translation_data(out, max_len=10, min_freq=1)
    assert data["stats"]["n_train"] == 80 and data["stats"]["n_test"] == 20
    assert "o\u00f9" in data["tgt_vocab"].stoi and "j'" in data["tgt_vocab"].stoi


def test_explicit_splits_are_deduplicated_against_train():
    tr = [(f"sentence number {i}", f"phrase num\u00e9ro {i}") for i in range(60)]
    te = [tr[3], tr[7], ("a new sentence", "une nouvelle phrase")] * 4
    data = T.build_translation_data({"train": tr, "val": te, "test": te}, max_len=10, min_freq=1)
    assert [e["tgt_tok"] for e in data["splits"]["test"]] == [["une", "nouvelle", "phrase"]]


def test_empty_eval_split_is_recarved_from_train():
    tr = [(f"w{i % 40}", f"m{i % 40}") for i in range(400)]
    data = T.build_translation_data({"train": tr, "val": tr[:50], "test": tr[:50]}, max_len=10, min_freq=1)
    assert data["stats"]["n_val"] >= 1 and data["stats"]["n_test"] >= 1 and data["stats"]["n_train"] >= 1


def test_translation_loader_anki_txt_json_and_file_pairs():
    d = _tmp()
    with open(os.path.join(d, "hin.txt"), "w", encoding="utf-8") as f:
        for e, h in zip(EN, HI):
            f.write(f"{e}\t{h}\tCC-BY 2.0 (France)\n")
    assert len(T.load_parallel(d)["all"]) == len(EN)
    d2 = _tmp()
    with open(os.path.join(d2, "train.jsonl"), "w", encoding="utf-8") as f:
        for e, h in zip(EN, HI):
            f.write(json.dumps({"translation": {"en": e, "hi": h}}, ensure_ascii=False) + "\n")
    assert len(T.load_parallel(d2)["train"]) == len(EN)
    d3 = _tmp()
    open(os.path.join(d3, "test.en"), "w", encoding="utf-8").write("\n".join(EN))
    open(os.path.join(d3, "test.hi"), "w", encoding="utf-8").write("\n".join(HI))
    assert T.load_parallel(d3)["test"][1] == (EN[1], HI[1])


def test_build_translation_data_splits_and_vocab():
    data = T.build_translation_data({"all": T.make_toy_en_hi(600, 0)["all"]}, max_len=12, min_freq=1, seed=0)
    st = data["stats"]
    assert st["n_train"] > st["n_val"] > 0 and st["n_test"] > 0
    keys = lambda k: {tuple(e["src_tok"]) + tuple(e["tgt_tok"]) for e in data["splits"][k]}
    assert not keys("train") & keys("test")                        # duplicates removed before splitting -> no leakage
    assert "शिक्षक" in data["tgt_vocab"].stoi or "लड़का" in data["tgt_vocab"].stoi


def _hf_rows():
    rows = []
    for cid, (doc, n) in enumerate([(3, 4), (5, 3)]):
        for t in range(n):
            rows.append({"conversation_id": f"c{cid}", "uid": f"user{t % 2 + 1}", "docIdx": t % 2, "wikiDocumentIdx": doc,
                         "utcTimestamp": f"2018-03-11 12:3{t}:00", "translation": {"en": f"turn {t} about the film", "hi_en": f"turn {t} film ke baare mein"},
                         "__split__": "train" if cid == 0 else "test"})
    return rows


def test_normalise_rows_hf_like_schema_with_external_docs():
    wiki = {"3": {"sections": {"0": "intro of film three", "1": "plot of film three"}, "full": "intro plot"},
            "5": {"sections": {"0": "intro five"}, "full": "intro five"}}
    convs = D.normalise_rows(_hf_rows(), wiki)
    assert [c["conv_id"] for c in convs] == ["c0", "c1"] and [len(c["turns"]) for c in convs] == [4, 3]
    assert convs[0]["turns"][0]["speaker"] == 0 and convs[0]["turns"][1]["speaker"] == 1
    assert convs[0]["doc_sections"]["1"] == "plot of film three" and convs[0]["split"] == "train"
    ex = D.make_examples(convs, hist_turns=2)
    e = next(x for x in ex if x["conv"] == "c0" and x["turn"] == 1)
    assert e["doc_tok"][:2] == ["plot", "of"]                      # section 1 = the section the responder was reading
    assert e["hist_tok"][0] in D.SPEAKER_TAGS


def test_nested_conversation_records_are_flattened():
    wiki = {"3": {"sections": {"0": "intro three"}, "full": "intro three"}}
    nested = [{"conversation_id": "a", "wikiDocumentIdx": 3,
               "history": [{"uid": "u1", "docIdx": 0, "translation": {"en": "hello", "hi_en": "namaste"}},
                           {"uid": "u2", "docIdx": 0, "translation": {"en": "hi there", "hi_en": "arre wah"}}]}]
    c = D.normalise_rows(nested, wiki)
    assert len(c) == 1 and [t["hi_en"] for t in c[0]["turns"]] == ["namaste", "arre wah"] and c[0]["doc_id"] == "3"
    parallel = [{"conversation_id": "b", "wikiDocumentIdx": 3, "uid": ["u1", "u2", "u1"], "docIdx": [0, 0, 0],
                 "translation": {"en": ["a", "b", "c"], "hi_en": ["aa", "bb", "cc"]}}]
    c = D.normalise_rows(parallel, wiki)
    assert [t["hi_en"] for t in c[0]["turns"]] == ["aa", "bb", "cc"] and [t["speaker"] for t in c[0]["turns"]] == [0, 1, 0]


def test_missing_documents_gives_actionable_error():
    try:
        D.normalise_rows(_hf_rows())
    except RuntimeError as e:
        assert "--docs" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_vocab_is_built_on_train_conversations_only():
    convs = D.make_toy_dog(60, 1)
    for c in convs:
        if c["split"] == "test":
            c["turns"][0]["hi_en"] += " zzzqqq"
    data = D.build_dialogue_data(convs, min_freq=1)
    assert "zzzqqq" not in data["vocab"].stoi
    assert data["stats"]["train"] > 0 and data["stats"]["test"] > 0


def test_code_mixing_proxy_and_buckets():
    eng = {"hi_en": "i really like this movie", "en": "i really like this movie"}
    hin = {"hi_en": "mujhe yeh film bahut pasand hai", "en": "i like this film very much"}
    assert D.english_share(eng) == 1.0 and D.english_share(hin) < 0.3
    assert D.mix_bucket(1.0) == "mostly English" and D.mix_bucket(0.1) == "mostly Hindi" and D.mix_bucket(None) == "unknown"


def test_tfidf_and_baselines():
    idx = TfidfIndex([["red", "apple"], ["green", "pear"], ["red", "car"]])
    assert idx.query(["green", "pear"], 1) == [1]
    tr = [{"hist_tok": ["<usr1>", "hello", "friend"], "resp_tok": ["hi", "there"], "doc_tok": []},
          {"hist_tok": ["<usr1>", "who", "directed", "it"], "resp_tok": ["rao", "did"], "doc_tok": []}]
    te = [{"hist_tok": ["<usr1>", "who", "directed", "it"], "doc_tok": ["it", "was", "shot", "in", "goa", ".", "rao", "directed", "it", "."]}]
    assert retrieval_baseline(tr, te)[0] == ["rao", "did"]
    assert extractive_baseline(te, tr)[0][:2] == ["rao", "directed"]


def test_no_repeat_ngram_helper():
    assert _banned([1, 2, 3, 1, 2], 3) == {3}
    assert _banned([1, 2], 3) == ()


def test_english_share_counts_devanagari_as_hindi():
    only_dev = {"hi_en": "मुझे यह फिल्म पसंद है", "en": "i like this film"}
    assert D.english_share(only_dev) == 0.0
    mixed = {"hi_en": "mujhe yeh movie pasand hai", "en": "i like this movie"}
    assert 0.0 < D.english_share(mixed) < 1.0
    assert D.english_share({"hi_en": "film", "en": "film"}) is None or D.english_share({"hi_en": "film", "en": "film"}) == 1.0


def _dog_rows(doc_ids, n_turns=4):
    rows = []
    for c, did in enumerate(doc_ids):
        for t in range(n_turns):
            rows.append({"wikiDocumentIdx": did, "docIdx": t % 2, "uid": t % 2 + 1, "utcTimestamp": f"2018-03-30T10:0{t}:00",
                         "date": f"d{c}", "translation": {"en": f"hello {t}", "hi_en": f"namaste doston {t}"}, "__split__": "train"})
    return rows


def _write_repo(root, names):
    wd = os.path.join(root, "datasets-CMU_DoG-master", "WikiData")
    os.makedirs(wd)
    os.makedirs(os.path.join(root, "datasets-CMU_DoG-master", "Conversations", "train"))
    json.dump({"x": "conversation file that must be ignored"}, open(os.path.join(root, "datasets-CMU_DoG-master", "Conversations", "train", "1.json"), "w"))
    for n in names:
        json.dump({"0": f"movie {n} naam hai", "1": f"yeh film {n} ke baare mein hai", "cast": ["a b", "c d"],
                   "introduction": "intro text", "movieName": f"Movie {n}"}, open(os.path.join(wd, f"{n}.json"), "w", encoding="utf-8"))
    return root


def test_docs_flag_accepts_repo_root_and_tolerates_float_ids():
    root = _write_repo(_tmp(), [3, 4])
    wiki = D.load_wiki_docs(root)                                  # repo ROOT, not the WikiData folder
    assert set(wiki) == {"3", "4"} and wiki["3"]["sections"]["1"].startswith("yeh film 3")
    rows = _dog_rows([3.0, 4.0])                                    # ids read as floats (NaN-containing column)
    convs = D.normalise_rows(rows, wiki)
    assert len(convs) == 2 and all(c["doc_full"] for c in convs)
    assert convs[0]["doc_sections"]["0"] == "movie 3 naam hai"


def test_doc_id_found_inside_file_when_file_names_differ():
    d = _tmp()
    json.dump({"wikiDocumentIdx": 7, "0": "alpha film text", "1": "second part"}, open(os.path.join(d, "movie_alpha.json"), "w"))
    wiki = D.load_wiki_docs(d)
    convs = D.normalise_rows(_dog_rows([7]), wiki)
    assert convs[0]["doc_full"].startswith("alpha film")


def test_unmatched_document_ids_give_a_diagnostic_and_partial_matches_are_filtered():
    root = _write_repo(_tmp(), [3])
    wiki = D.load_wiki_docs(root)
    try:
        D.normalise_rows(_dog_rows([11, 12]), wiki)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "'11'" in str(e) and "'3'" in str(e)                # shows both id lists
    convs = D.normalise_rows(_dog_rows([3, 11]), wiki)              # one matches: the unmatched conversation is dropped, not used with no document
    assert len(convs) == 1


def test_consecutive_conversations_do_not_merge_across_splits():
    rows = _dog_rows([3, 3], n_turns=2)
    for r in rows[2:]:
        r["__split__"] = "val"
    convs = D.normalise_rows(rows, {"3": {"sections": {"0": "doc text"}, "full": "doc text"}})
    assert sorted(c["split"] for c in convs) == ["train", "val"]


MOVIES = {0: ("avengers", "marvel superhero film directed by whedon starring downey"), 1: ("titanic", "ship iceberg romance film directed by cameron starring dicaprio"),
          2: ("gladiator", "roman empire arena film directed by scott starring crowe"), 3: ("inception", "dream heist film directed by nolan starring dicaprio")}


def _real_schema_rows(id_shift=0):
    """Rows shaped like the real festvox/cmu_hinglish_dog parquet files: no conversation id, session fields repeat on every turn."""
    rows = []
    for c, (did, (name, blurb)) in enumerate(MOVIES.items()):
        for conv in range(2):                                          # two back-to-back conversations about the SAME movie
            for t in range(4):
                en = f"do you like {name}? it is a {blurb}" if t % 2 == 0 else f"yes {name} is great, {blurb}"
                rows.append({"date": f"2017-12-0{c + 1}T15:4{conv}:50Z", "docIdx": t % 2, "uid": f"user{t % 2 + 1}",
                             "utcTimestamp": f"2017-12-0{c + 1}T15:4{conv}:5{t}Z", "uid1LogInTime": f"2017-12-0{c + 1}T15:4{conv}:50Z",
                             "uid1LogOutTime": f"2017-12-0{c + 1}T16:0{conv}:00Z", "user2_id": f"USR{c}{conv}",
                             "wikiDocumentIdx": did + id_shift, "translation": {"en": en, "hi_en": "arre yaar " + en}, "__split__": "train"})
    return rows


def _movie_docs(keys_shift=0):
    return {str(did + keys_shift): {"sections": {"0": f"{n} {b}", "1": f"more about {n} {b}"}, "full": f"{n} {b} more about {n}"}
            for did, (n, b) in MOVIES.items()}


def test_session_fields_give_exact_conversations():
    convs = D.normalise_rows(_real_schema_rows(), _movie_docs())
    assert len(convs) == 8 and all(len(c["turns"]) == 4 for c in convs)        # time/doc heuristic would have merged the pairs


def test_alignment_check_keeps_correct_ids_and_remaps_shifted_ones():
    good = D.verify_doc_alignment(D.normalise_rows(_real_schema_rows(), _movie_docs()), _movie_docs())
    assert all(MOVIES[int(c["doc_id"])][0] in c["doc_full"] for c in good)
    wiki = _movie_docs(keys_shift=1)                                           # document files numbered 1..4, dialogues use 0..3
    wrong = D.normalise_rows(_real_schema_rows(), wiki)                        # dialogues 0..3 vs files 1..4: ids 1..3 pair with the WRONG film
    fixed = D.verify_doc_alignment(wrong, wiki)
    assert all(MOVIES[int(c["doc_id"])][0] in c["doc_full"] for c in fixed if c["doc_id"] != "0")


def test_fetch_extracts_only_wikidata_with_short_paths():
    import io
    import unittest.mock as mock
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in range(3):
            zf.writestr(f"datasets-CMU_DoG-master/WikiData/{n}.json", json.dumps({"0": f"film {n} text", "1": "more"}))
        zf.writestr("datasets-CMU_DoG-master/Conversations/test/" + "a" * 40 + ".json", json.dumps({"x": "ignored"}))
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return buf.getvalue()
    d = _tmp()
    with mock.patch("urllib.request.urlopen", lambda *a, **k: Resp()):
        wiki = D.fetch_cmu_dog_wikidata(d)
    assert set(wiki) == {"0", "1", "2"} and sorted(os.listdir(os.path.join(d, "cmu_dog_wikidata"))) == ["0.json", "1.json", "2.json"]


def test_add_english_doubles_train_examples_only():
    convs = D.make_toy_dog(n_conv=60, seed=1)
    for c in convs:
        for tr in c["turns"]:
            tr.setdefault("en", "the english version of " + tr["hi_en"])
    convs = D.assign_splits(convs, 0)
    base = D.build_dialogue_data(convs, None, None, 0, 1, 2000)
    aug = D.build_dialogue_data(convs, None, None, 0, 1, 2000, add_english=True)
    assert aug["stats"]["train"] > base["stats"]["train"] * 1.5
    assert aug["stats"]["val"] == base["stats"]["val"] and aug["stats"]["test"] == base["stats"]["test"]
    assert {e["lang"] for e in aug["examples"]["train"]} == {"hi_en", "en"} and {e["lang"] for e in aug["examples"]["test"]} == {"hi_en"}


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("PASS", k)
