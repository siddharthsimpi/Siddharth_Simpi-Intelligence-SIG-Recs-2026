# Review notes: changes made to the solution

Checked every sub-task against the README requirements (basic LSTM enc-dec + BLEU; attention + decoding comparison;
two-source grounded Hinglish dialogue with from-scratch embeddings, baselines, BLEU/ROUGE and code-mix analysis).
The architecture, training, decoding and evaluation code were already correct and all original tests passed.
The English / Hindi handling had real bugs, now fixed:

1. `str.isalpha()` is False for Devanagari words that contain a vowel sign or virama (e.g. "है", "हिन्दी").
   The grounding metrics (`grounding_overlap`, `doc_word_hit`) therefore silently ignored Hindi words
   (NaN / 0 for Devanagari replies). Fixed with a script-aware `tokenize.is_wordlike` and `metrics.content_words`
   (Latin >= 4 chars, Devanagari >= 3 code points) plus a Devanagari stop-word list.
2. The danda "।" was classified as a Devanagari word by `script_of`; it is now punctuation. This also corrected the
   Devanagari-token share statistic.
3. Zero-width joiner / non-joiner inside Devanagari words split one word into several tokens
   ("क्‍ष" -> "क्", ZWJ, "ष"). They are now removed, so spelling variants map to one token.
4. Code-mix bucketing (`english_share`) ignored Devanagari tokens, so Devanagari-only turns became "unknown". They now count as
   Hindi. The bucket label "mostly Hindi (romanised)" is now "mostly Hindi".
5. The sub-task 3 report no longer hard-codes "the hi_en field is romanised"; it states this from the measured Devanagari share.
6. Added regression tests for all of the above (tests/test_text.py, tests/test_data.py).

## Round 2: "no grounding-document text" / loader crashes (sub-task 3)
- `--docs` now accepts the CMU DoG repo root, the WikiData folder or a JSON file; conversation files in the repo are ignored.
- Document ids are matched tolerantly (4 == 4.0 == "4"), and also by an id stored inside each document file.
- Nested document JSON (lists / dicts such as cast, scenes) is flattened to text; digit keys remain the sections.
- Real error messages are printed for every failed Hugging Face route (the old code printed only the exception type), and two more routes
  were added (trust_remote_code, parquet branch).
- A mismatch now prints the dialogue document ids next to the loaded ids; conversations without a document are dropped instead of silently used.
- Consecutive conversations on the same document no longer merge across splits.
- `inspect_dataset.py` reports id matching and fields holding document text; HOW_TO_RUN.md has a troubleshooting section; 4 new tests.

## Second round: sub-task 3 data loading (found on a real Windows run)
7. `fetch_cmu_dog_wikidata` extracted the whole CMU DoG repository and crashed on Windows (path > 260 characters). It now extracts only the
   `WikiData` JSON files, flat, into `data/cmu_dog_wikidata` (with a temp-folder fallback and extended-length paths) and caches them.
8. The real Hugging Face rows have no conversation id; conversations are now grouped exactly from the session fields
   (`wikiDocumentIdx`, `uid1LogInTime`, `uid1LogOutTime`, `user2_id`) instead of a time-gap heuristic. The heuristic also no longer merges across splits.
9. Document loading is tolerant: `--docs` may be the repo root, the WikiData folder, or a JSON file; ids match as 7 / 7.0 / "7"; ids stored inside the
   files are also used; conversations without a document are dropped instead of being trained with an empty document.
10. New document-alignment check: compares each conversation's text with every loaded document and re-maps (loudly) if the ids are shifted.
11. Clearer errors: real `datasets` exception text, extra download routes, `inspect_dataset.py` lists dialogue ids vs document ids.

## Third round: collapsed generations on the real data
12. `generate` now accepts `min_len` (greedy, beam and sampling); `run_subtask3.py --min-len` (default 4) stops 1-3 word replies.
13. `--add-english` trains on the English side of each training turn as extra data (val/test unchanged).
14. The report states when neural output has collapsed to generic replies instead of concluding the document is "used weakly".

## Fourth round: full re-audit against the task READMEs
- Re-read the four task READMEs and re-checked every requirement (from-scratch LSTM enc-dec + BLEU; attention + decoding comparison;
  two-source grounded Hinglish dialogue, from-scratch embeddings, baselines, BLEU/ROUGE, code-mix analysis).
- Verified: all unit tests pass; `run_all.py --quick --toy` passes end to end; sub-tasks 1-2 run on English-French CSV files named
  `en_fr_train/val/test.csv`; sub-task 3 runs on Hugging Face-style session rows + a WikiData folder with every `--attn` option,
  `--add-english`, `--ablate-emb`, `--beam 0`; beam search with width 1 reproduces greedy decoding exactly (attention and no-attention models).
- Fixed: the sub-task 1 report printed two empty bullet points when the length / rare-word buckets were too small to compare;
  it now prints only the observations that exist (or a one-line note).
