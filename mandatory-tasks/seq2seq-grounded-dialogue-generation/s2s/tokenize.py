"""Tokenisation and vocabularies for English, Devanagari Hindi and romanised/code-mixed Hinglish.

Why custom: Python's `\\w` does NOT match Devanagari vowel signs (matras) or the virama, so a naive `\\w+` regex cuts Hindi
words into pieces; and social-chat Hinglish is full of elongations ("yaaaar"), URLs, emojis and inconsistent spelling.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]

_DEV = r"[\u0900-\u0963\u0966-\u097F]+"           # Devanagari letters + matras + virama (danda U+0964/0965 excluded)
_LETTERS = "a-z0-9\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f"     # ASCII + accented Latin letters (\u00e9 \u00e7 \u0153 ...)
_ELISION = rf"(?:qu|jusqu|lorsqu|puisqu|[ldjmtsnc])'(?=[{_LETTERS}])"   # French l' d' j' qu' ... split off as own token
_LAT = rf"[{_LETTERS}]+(?:'[{_LETTERS}]+)?"        # latin word, optional apostrophe part (don't, ye'll)
_TOKEN_RE = re.compile(f"{_DEV}|{_ELISION}|{_LAT}|<[a-z0-9_]+>|\\S")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# zero-width joiner / non-joiner next to Devanagari are invisible spelling variants ("क्‍ष" vs "क्ष"): drop them so one word = one token
_ZW_DEV_RE = re.compile("(?<=[\u0900-\u097f])[\u200c\u200d]|[\u200c\u200d](?=[\u0900-\u097f])")
_ELONG_RE = re.compile(r"([a-z])\1{2,}")           # yaaaar -> yaar (3+ repeats collapse to 2, see normalize)


def normalize(text: str, lower: bool = True, collapse_elongation: bool = True) -> str:
    text = unicodedata.normalize("NFC", text).replace("\u2019", "'").replace("\u2018", "'")
    text = _ZW_DEV_RE.sub("", text)
    if lower:
        text = text.lower()
    text = _URL_RE.sub(" <url> ", text)
    if collapse_elongation:
        text = _ELONG_RE.sub(r"\1\1", text)        # 3+ repeats -> 2 ("sooooo" -> "soo"): keeps emphasis, cuts vocab
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, lower: bool = True) -> list[str]:
    return _TOKEN_RE.findall(normalize(text, lower))


def script_of(tok: str) -> str:
    """'dev' (Devanagari), 'lat' (latin/number word) or 'other' (punctuation, emoji, tags)."""
    if "\u0900" <= tok[0] <= "\u097f" and tok[0] not in "\u0964\u0965":      # danda / double danda are punctuation, not words
        return "dev"
    if tok[0].isalnum():
        return "lat"
    return "other"


def is_wordlike(tok: str) -> bool:
    """True for a real word in Latin OR Devanagari script. (str.isalpha() is False for Devanagari words that contain a vowel
    sign / virama, e.g. 'है', 'हिन्दी', so it must not be used to detect words in Hindi text.)"""
    return script_of(tok) in ("dev", "lat") and not tok.startswith("<") and not tok.isdigit()


_NO_SPACE_BEFORE = set(".,!?;:%)]}\u0964\u0965")
_NO_SPACE_AFTER = set("([{$")


def detokenize(tokens: list[str]) -> str:
    out = ""
    for t in tokens:
        if not out:
            out = t
        elif t in _NO_SPACE_BEFORE or out[-1] in _NO_SPACE_AFTER or out.endswith("'"):
            out += t
        else:
            out += " " + t
    return out


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.unk, self.bos, self.eos = (self.stoi[PAD], self.stoi[UNK], self.stoi[BOS], self.stoi[EOS])

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, token_lists, min_freq: int = 2, max_size: int | None = None, extra_specials=()) -> "Vocab":
        counts = Counter(t for toks in token_lists for t in toks)
        specials = SPECIALS + [s for s in extra_specials if s not in SPECIALS]
        words = [w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                 if c >= min_freq and w not in specials]
        if max_size is not None:
            words = words[: max(0, max_size - len(specials))]
        v = cls(specials + words)
        v.counts = counts
        return v

    def encode(self, tokens, add_bos=False, add_eos=False) -> list[int]:
        ids = [self.stoi.get(t, self.unk) for t in tokens]
        return ([self.bos] if add_bos else []) + ids + ([self.eos] if add_eos else [])

    def decode(self, ids, strip_special: bool = True) -> list[str]:
        toks = []
        for i in ids:
            i = int(i)
            if i == self.eos:
                break
            if strip_special and i in (self.pad, self.bos):
                continue
            toks.append(self.itos[i])
        return toks

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def oov_rate(self, token_lists) -> float:
        n = u = 0
        for toks in token_lists:
            for t in toks:
                n += 1
                u += t not in self.stoi
        return u / max(n, 1)


def pad_batch(seqs: list[list[int]], pad: int, min_len: int = 1):
    """list of id lists -> (ids (B,L) int64, lengths (B,), mask (B,L) float)."""
    import numpy as np
    L = max(max((len(s) for s in seqs), default=0), min_len)
    ids = np.full((len(seqs), L), pad, dtype=np.int64)
    mask = np.zeros((len(seqs), L), dtype=np.float32)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, : len(s)] = 1.0
    return ids, np.array([len(s) for s in seqs]), mask
