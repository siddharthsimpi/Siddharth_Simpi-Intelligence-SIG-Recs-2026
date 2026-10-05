# Shopee Product Matching — Parts A, B, C and Finale

Code for the mandatory task (dataset exploration → text matching → image matching → multimodal system).

```
shopee-solution/
├── README.md, requirements.txt, run_all.sh, run_smoke.sh
├── shopee_match/          # shared library (data, embeddings+cache, retrieval, metrics, benchmark pairs, plots)
├── tools/                 # build_notebooks.py, make_synthetic_data.py
├── PartA/  PartB/  PartC/ # notebook.py/.ipynb + README.md + results/
└── Finale/                # notebook, src/, results/, README.md, report.pdf (generated)
```

## 1. Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.10+
pip install -r requirements.txt
```
Download the Kaggle data (needs a Kaggle account / API token) and unzip so that `train.csv` and `train_images/` exist:
```bash
pip install kaggle
kaggle competitions download -c shopee-product-matching -p data
mkdir -p data/shopee-product-matching && unzip -q data/shopee-product-matching.zip -d data/shopee-product-matching
# expected: data/shopee-product-matching/train.csv  and  data/shopee-product-matching/train_images/*.jpg
```
(Accept the competition rules on kaggle.com first.) Use another location with `export SHOPEE_DATA_DIR=/path/to/shopee-product-matching`.

## 2. Verify the pipeline (1–2 min, no GPU, no downloads)
```bash
bash run_smoke.sh
```
## 3. Real run
```bash
# optional fast dev run on 3000 product groups (~9k listings):  export SHOPEE_MAX_GROUPS=3000
bash run_all.sh                       # builds + executes the 4 notebooks in order, stores outputs in the .ipynb files
# or run one part:  cd PartB && python notebook.py
```
Order matters only loosely: Part C/Finale reuse cached embeddings in `cache/`; Finale additionally reads Parts B/C result CSVs for context if present.
Rough cost on the full data (34k images): ResNet-50 + CLIP embedding ≈ 10–15 min on a single GPU (≈ 1 h+ on CPU); everything else ≈ 10–20 min.

## Windows (Command Prompt) equivalents
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONUTF8=1
run_smoke.bat
:: real data: set SHOPEE_DATA_DIR=C:\data\shopee-product-matching  (keep the 34k images OUTSIDE OneDrive)
:: quick trial: set SHOPEE_MAX_GROUPS=3000
run_all.bat
```
(PowerShell: activate with `.venv\Scripts\Activate.ps1`, set variables with `$env:NAME="value"`.)

## 4. Inference with the final model
```bash
python Finale/src/predict.py --csv data/shopee-product-matching/train.csv --images data/shopee-product-matching/train_images --out predictions.csv
```

## Design choices worth knowing (you will be asked about them)
* **Group-wise split** (60/20/20 by `label_group`) – products never appear in two splits; thresholds/weights/model selection use val only, tables report test.
* **Two metrics**: pair-level (AUC/AP/F1 on a fixed benchmark with random, text-hard and image-hard negatives) and the Kaggle per-listing F1 over match sets.
* **Same preprocessing for every vision backbone**: resize whole image to 224×224 (no centre crop).
* **TF-IDF is fit on all titles** (unsupervised/transductive; no labels used) – listed as a limitation.
* The benchmark and the retrieval protocol are identical across Parts B, C and Finale, so numbers are directly comparable.

## External resources
scikit-learn; `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; OpenAI CLIP ViT-B/32 via HuggingFace `transformers`; torchvision ResNet-50 / EfficientNet-B0; optional faiss; Kaggle "Shopee – Price Match Guarantee" dataset & metric.

## Status of this code
Everything except the neural-model code paths was executed end-to-end on a synthetic dataset (smoke test). The real encoders (ResNet/EfficientNet/CLIP/SBERT) could not be run in the authoring environment (no network/GPU), so **run `run_smoke.sh` first, then the real run, and read each notebook's Observations cells critically.**
