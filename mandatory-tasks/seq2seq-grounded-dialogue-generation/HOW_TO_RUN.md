# How to run (Windows PowerShell / cmd; identical on macOS/Linux)

Python 3.9+ is enough. Everything is NumPy (no PyTorch / TensorFlow). From this folder:

```
pip install -r requirements.txt
pip install datasets pyarrow gdown          # optional but recommended: lets the scripts download the datasets
python run_tests.py                         # unit tests, expect ALL TESTS PASSED (about 1 minute)
python run_all.py --quick --toy             # smoke test of the whole pipeline on SYNTHETIC data (about 5 minutes)
```
The `--toy` run only proves the code works end to end. It produces results marked with a WARNING; do not report them.

## Real runs (one command per sub-task)
```
cd subtask1
python run_subtask1.py          # basic encoder-decoder, BLEU, failure modes
cd ..\subtask2
python run_subtask2.py          # attention (Bahdanau, Luong) + decoding strategies; re-uses sub-task 1's model
cd ..\subtask3
python inspect_dataset.py       # look at the Hinglish DoG fields first (recommended)
python run_subtask3.py          # grounded vs ungrounded vs retrieval baselines
cd ..
python make_final_report.py     # builds FINAL_REPORT.md (the short 2-3 page write-up)
```
Order matters for 1 -> 2 (sub-task 2 re-uses the sub-task 1 model when the settings are identical, otherwise retrains it).

Rough time on one laptop CPU core (each trained model is capped by `--max-seconds`, default 1500 s for translation, 1800 s for dialogue):
sub-task 1 about 25 min, sub-task 2 about 1 h, sub-task 3 about 1-1.5 h. Use `--quick` for a 1-2 minute check, `--max-train 5000 --epochs 5` for a shorter real run.

## Data
* **Sub-tasks 1-2 (translation):** put the files of the course Google Drive folder into `data\translation\` (CSV / TSV / TXT / JSON / parallel `.en`+`.fr` files all work),
  or run with `--data <folder>`. The language pair is inferred from the file names (`en_fr_train.csv` -> English to French); override with `--src-lang en --tgt-lang fr`.
  Files named train / val / test are used as the splits. If your columns are not recognised, pass `--src-col NAME --tgt-col NAME`.
  If the folder is empty the script tries `gdown` for the Drive folder and then downloads the Tatoeba/Anki English-Hindi pairs.
* **Sub-task 3:** downloads `festvox/cmu_hinglish_dog` from Hugging Face automatically (needs internet; `pip install datasets` is the most reliable route).
  The dialogue rows reference Wikipedia documents by index; if the rows do not contain the text, the script fetches the original CMU DoG repository for the
  documents. If that fails, pass `--docs <json file or folder>` with the document text. Use `--data <folder>` for locally downloaded files.
  `python inspect_dataset.py` prints the fields it finds, so format problems are easy to diagnose.

## Where the results are
`subtask1\results\report.md`, `subtask2\results\report.md`, `subtask3\results\report.md` (tables, figures, example outputs, analysis), plus `FINAL_REPORT.md`.
In VS Code press Ctrl+Shift+V to preview a `.md` file.

## Useful options
`--quick`, `--seed`, `--max-train N`, `--max-seconds S`, `--epochs E`; sub-task 2: `--with-dot`; sub-task 3: `--attn {bahdanau,luong_general,luong_dot,none}`,
`--ablate-emb` (also train with random instead of skip-gram embeddings), `--beam 0` (skip beam decoding), `--no-sgns`.

## If sub-task 3 says "no grounding-document text" or crashes while loading
1. Run `python inspect_dataset.py` (in `subtask3`) and read what it prints: the first raw records, the document ids the dialogues use, and whether any field holds document text.
2. The Hugging Face rows only reference the Wikipedia documents by index (`wikiDocumentIdx`); the text comes from the original repository.
   Download https://github.com/festvox/datasets-CMU_DoG (green "Code" button -> Download ZIP), unzip it, then run
   `python run_subtask3.py --docs path\to\datasets-CMU_DoG-master` (the repo root or its `WikiData` folder both work). `inspect_dataset.py --docs <same path>` tells you whether the ids match.
3. If `datasets` fails (newer versions refuse dataset scripts), the log shows the exact error and the code tries other routes automatically;
   `pip install "datasets<4"` is the quickest fix. With no internet, download the files by hand and use `--data <folder>`.
4. Still stuck: send the complete terminal output of `python inspect_dataset.py` (and of the failing command).

### Windows: "FileNotFoundError ... data\\cmu_dog_repo\\...\\Conversations\\test\\<hash>.json"
This is Windows' 260-character path limit (deep OneDrive/Desktop project folder + the CMU DoG repo's long file names). The loader now extracts
only the `WikiData` documents into `data\\cmu_dog_wikidata` (short paths) and never unpacks the `Conversations` folder. You can delete the
half-extracted `data\\cmu_dog_repo` folder left over from the failed run. Moving the project to a short path such as `C:\\s2s` also avoids the problem.

### Sub-task 3: replies are all "kya aap" / scores near zero
The real Hinglish DoG training set is tiny (about 8,000 reply turns, roughly 80k words), so a from-scratch model tends to collapse to a few generic replies.
The report flags this automatically ("degenerate generation"). Two options help, and the report records which were used:
`--min-len 4` (default; forbids 1-3 word replies) and `--add-english` (also trains on the English version of every training turn; validation and test stay Hinglish).
Suggested run: `python subtask3\\run_subtask3.py --add-english --max-vocab 4000 --min-freq 3 --dropout 0.4 --max-seconds 900`.
Report the numbers you get, including the retrieval baseline and the sampled-decoding rows, and describe the collapse honestly in the write-up.
