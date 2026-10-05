"""Global configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SHOPEE_DATA_DIR", ROOT / "data" / "shopee-product-matching"))
CACHE_DIR = Path(os.environ.get("SHOPEE_CACHE_DIR", ROOT / "cache"))

SEED = 42
# 0 = use the full dataset. N > 0 = keep N random product groups (fast dev runs).
MAX_GROUPS = int(os.environ.get("SHOPEE_MAX_GROUPS", "0"))
# SMOKE=1 swaps every neural model for a tiny deterministic stand-in (no downloads, no GPU).
SMOKE = os.environ.get("SHOPEE_SMOKE", "0") == "1"

SPLIT_FRACS = (0.6, 0.2, 0.2)   # train / val / test, split BY product group (no leakage)
TOPK = 50                        # neighbours retrieved per listing (max group size in data is 51)

# Embedding names used by the notebooks (real model vs. smoke stand-in)
if SMOKE:
    SBERT, CLIP_TXT = "hash_text_256", "hash_text_128"
    RESNET, EFFNET, CLIP_IMG = "tiny_pixels_16", "tiny_pixels_8", "tiny_pixels_12"
else:
    SBERT, CLIP_TXT = "sbert", "clip_text"
    RESNET, EFFNET, CLIP_IMG = "resnet50", "effnet_b0", "clip_img"


def results_dir(part: str) -> Path:
    d = ROOT / part / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d
