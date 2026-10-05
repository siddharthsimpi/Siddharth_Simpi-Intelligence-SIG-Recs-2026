import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from s2s import metrics as M  # noqa: E402
from s2s import tokenize as T  # noqa: E402


def test_devanagari_words_are_not_split():
    toks = T.tokenize("मुझे यह फिल्म बहुत पसंद है।")
    assert toks == ["मुझे", "यह", "फिल्म", "बहुत", "पसंद", "है", "।"], toks     # naive \w+ would cut at matras


def test_french_accents_and_elision():
    assert T.tokenize("L'\u00e9t\u00e9 dernier, qu'il a d\u00e9j\u00e0 fran\u00e7ais.") == ["l'", "\u00e9t\u00e9", "dernier", ",", "qu'", "il", "a", "d\u00e9j\u00e0", "fran\u00e7ais", "."]
    assert T.tokenize("Je n\u2019ai pas d\u2019argent") == ["je", "n'", "ai", "pas", "d'", "argent"]
    assert T.tokenize("I don't know, she's here") == ["i", "don't", "know", ",", "she's", "here"]
    assert T.detokenize(T.tokenize("l'homme qu'il a vu !")) == "l'homme qu'il a vu!"
    assert T.script_of("\u00e9t\u00e9") == "lat"


def test_hinglish_normalisation():
    toks = T.tokenize("Yaaaar!! Ye movie sooo accha hai... https://x.com/a 😀")
    assert "yaar" in toks and "<url>" in toks and "😀" in toks and "!" in toks
    assert T.tokenize("Don't worry") == ["don't", "worry"]
    assert T.script_of("फिल्म") == "dev" and T.script_of("movie") == "lat" and T.script_of("!") == "other"


def test_vocab_roundtrip_and_oov():
    v = T.Vocab.build([["a", "b", "a"], ["a", "c"]], min_freq=2)
    assert v.itos[:4] == T.SPECIALS and "a" in v.stoi and "c" not in v.stoi
    ids = v.encode(["a", "c"], add_bos=True, add_eos=True)
    assert ids == [v.bos, v.stoi["a"], v.unk, v.eos]
    assert v.decode(ids) == ["a", T.UNK]
    assert abs(v.oov_rate([["a", "c"]]) - 0.5) < 1e-9


def test_detokenize():
    assert T.detokenize(["hello", ",", "world", "!"]) == "hello, world!"


def test_bleu_known_values():
    ref = "the cat is on the mat".split()
    assert abs(M.corpus_bleu([ref], [ref])["bleu"] - 100) < 1e-6
    assert M.corpus_bleu([ref], ["dog dog dog".split()])["bleu"] == 0.0
    # brevity penalty: hypothesis half as long with perfect precision
    r = "a b c d e f g h".split(); h = "a b c d".split()
    out = M.corpus_bleu([r], [h], smooth=False)
    assert abs(out["bp"] - math.exp(1 - 8 / 4)) < 1e-9 and abs(out["bleu"] - 100 * math.exp(-1)) < 1e-6


def test_bleu_matches_nltk_if_available():
    try:
        from nltk.translate.bleu_score import corpus_bleu as nb
    except Exception:
        return
    refs = [["the cat sat on the mat".split()], ["there is a book on the table".split()]]
    hyps = ["the cat is on the mat".split(), "there is a book on a table".split()]
    mine = M.corpus_bleu([r[0] for r in refs], hyps, smooth=False)["bleu"] / 100
    assert abs(mine - nb(refs, hyps)) < 1e-9


def test_rouge_distinct_overlap():
    assert M.rouge_n("a b c".split(), "a b c".split(), 1) == 1.0
    assert abs(M.rouge_l("a b c d".split(), "a c d".split()) - 2 * 1 * 0.75 / 1.75) < 1e-9
    assert M.distinct_n([["a", "a", "b"]], 1) == 2 / 3
    assert M.grounding_overlap(["director", "spielberg", "the"], ["spielberg", "director", "x"]) == 1.0


def test_hindi_words_count_as_words_and_grounding_works_on_devanagari():
    assert T.is_wordlike("है") and T.is_wordlike("हिन्दी") and T.is_wordlike("movie") and not T.is_wordlike("।") and not T.is_wordlike("<url>")
    doc = T.tokenize("यह फिल्म मुंबई में बनी थी।")
    assert M.content_words(T.tokenize("फिल्म मुंबई में बनी")) == ["फिल्म", "मुंबई", "बनी"]
    assert M.grounding_overlap(T.tokenize("फिल्म मुंबई में बनी"), doc) == 1.0          # was nan when isalpha() dropped Hindi words
    mixed = M.grounding_overlap(T.tokenize("the film मुंबई में bani"), T.tokenize("film मुंबई"))
    assert 0.0 < mixed <= 1.0


def test_zero_width_joiners_do_not_split_hindi_words():
    assert T.tokenize("क्\u200dष और ज्ञान") == ["क्ष", "और", "ज्ञान"]
    assert T.tokenize("नमस्ते\u200c दोस्त") == ["नमस्ते", "दोस्त"]


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            v()
            print("PASS", k)
