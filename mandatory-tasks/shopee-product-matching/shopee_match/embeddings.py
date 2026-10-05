"""Embedding extraction with on-disk caching.

Names
-----
Text : tfidf_word | tfidf_word12 | tfidf_char | sbert | clip_text | hash_text_<dim> (smoke)
Image: phash_bits | resnet50 | effnet_b0 | clip_img | tiny_pixels_<side> (smoke)

Dense embeddings are cached RAW (un-normalised) so that experiments can compare
normalisation / similarity choices; use `get_embedding(..., normalize=True)` for unit vectors.
TF-IDF matrices are sparse and already L2-normalised.
"""
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from PIL import Image

from . import config as C
from .text import normalize_title, tfidf_matrix

SBERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CLIP_MODEL = "openai/clip-vit-base-patch32"
IMG_SIZE = 224


def l2norm(x, eps=1e-12):
    if sp.issparse(x):
        return x
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def device_name():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- images
def load_rgb(path, size=None):
    """Open an image as RGB. Broken / missing files become a mid-grey image (never crash)."""
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        im = Image.new("RGB", size or (IMG_SIZE, IMG_SIZE), (128, 128, 128))
    return im


def _torch_image_embed(paths, model_fn, mean, std, batch_size=128, num_workers=4):
    """Generic batched image -> vector loop. Same preprocessing for every backbone:
    resize the WHOLE image to 224x224 (no centre-crop, so objects at the border are not cut)."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
    from tqdm.auto import tqdm

    tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(mean, std)])

    class DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return tf(load_rgb(paths[i]))

    dev = device_name()
    model = model_fn().to(dev).eval()
    if os.name == "nt":      # Windows uses 'spawn': a locally-defined Dataset cannot be pickled and the notebooks
        num_workers = 0      # have no __main__ guard, so worker processes must be disabled there.
    dl = DataLoader(DS(), batch_size=batch_size, num_workers=num_workers)
    out = []
    with torch.no_grad():
        for x in tqdm(dl, desc="images"):
            x = x.to(dev)
            if dev == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    f = model(x)
            else:
                f = model(x)
            out.append(f.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


_IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_CLIPN = ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])


def embed_resnet50(paths):
    import torch.nn as nn
    import torchvision

    def make():
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()          # 2048-d global-average-pooled features
        return m
    return _torch_image_embed(paths, make, *_IMNET)


def embed_effnet_b0(paths):
    import torch.nn as nn
    import torchvision

    def make():
        m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Identity()  # 1280-d
        return m
    return _torch_image_embed(paths, make, *_IMNET)


def embed_clip_image(paths):
    import torch.nn as nn
    from transformers import CLIPModel

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = CLIPModel.from_pretrained(CLIP_MODEL)

        def forward(self, x):
            f = self.m.get_image_features(pixel_values=x)
            return getattr(f, "pooler_output", f)   # robust to API differences between versions
    return _torch_image_embed(paths, Wrap, *_CLIPN)


def embed_tiny_pixels(paths, side):
    """SMOKE stand-in for a CNN: per-channel colour histogram of non-background pixels."""
    out = []
    for p in paths:
        a = np.asarray(load_rgb(p, (side, side)).resize((side, side)), dtype=np.float32).reshape(-1, 3)
        a = a[a.min(1) < 200] if (a.min(1) < 200).any() else a
        h = [np.histogram(a[:, c], bins=16, range=(0, 256))[0] for c in range(3)]
        out.append(np.concatenate(h).astype(np.float32) / len(a))
    return np.stack(out)


# --------------------------------------------------------------------------- text
def embed_sbert(titles, batch_size=128):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(SBERT_MODEL, device=device_name())
    return m.encode(list(titles), batch_size=batch_size, show_progress_bar=True,
                    convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)


def embed_clip_text(titles, batch_size=256):
    import torch
    from transformers import CLIPModel, CLIPTokenizer
    dev = device_name()
    tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(dev).eval()
    titles, out = list(titles), []
    with torch.no_grad():
        for a in range(0, len(titles), batch_size):
            enc = tok(titles[a:a + batch_size], padding=True, truncation=True, max_length=77,
                      return_tensors="pt").to(dev)
            f = model.get_text_features(**enc)
            out.append(getattr(f, "pooler_output", f).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def embed_hash_text(titles, dim):
    """SMOKE stand-in: hashed char n-grams."""
    from sklearn.feature_extraction.text import HashingVectorizer
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4), n_features=dim,
                           alternate_sign=False, norm=None)
    return hv.transform(list(titles)).toarray().astype(np.float32)


# --------------------------------------------------------------------------- public API
def _compute(df, name):
    titles = [normalize_title(t) for t in df["title"]]
    if name == "tfidf_word":
        return tfidf_matrix(titles, "word", (1, 1))
    if name == "tfidf_word12":
        return tfidf_matrix(titles, "word", (1, 2))
    if name == "tfidf_char":
        return tfidf_matrix(titles, "char_wb", (2, 5), min_df=2)
    if name == "sbert":
        return embed_sbert(df["title"].astype(str))
    if name == "clip_text":
        return embed_clip_text(df["title"].astype(str))
    if name.startswith("hash_text_"):
        return embed_hash_text(titles, int(name.split("_")[-1]))
    if name == "phash_bits":      # 64 bits as +-1 vector: cosine = 1 - 2*Hamming/64
        b = np.stack([np.unpackbits(np.frombuffer(int(h, 16).to_bytes(8, "big"), dtype=np.uint8))
                      for h in df["image_phash"]]).astype(np.float32)
        return (b * 2 - 1) / 8.0
    paths = df["path"].tolist()
    if name == "resnet50":
        return embed_resnet50(paths)
    if name == "effnet_b0":
        return embed_effnet_b0(paths)
    if name == "clip_img":
        return embed_clip_image(paths)
    if name.startswith("tiny_pixels_"):
        return embed_tiny_pixels(paths, int(name.split("_")[-1]))
    raise ValueError(f"unknown embedding '{name}'")


def get_embedding(df, name, tag, normalize=True):
    """Return the embedding matrix aligned with df's row order (computed once, then cached)."""
    d = C.CACHE_DIR / tag
    d.mkdir(parents=True, exist_ok=True)
    f_npy, f_npz, f_meta = d / f"{name}.npy", d / f"{name}.npz", d / f"{name}.json"
    if f_npz.exists():
        E = sp.load_npz(f_npz).tocsr()
    elif f_npy.exists():
        E = np.load(f_npy)
    else:
        t0 = time.time()
        E = _compute(df, name)
        secs = time.time() - t0
        if sp.issparse(E):
            sp.save_npz(f_npz, E)
        else:
            np.save(f_npy, E)
        json.dump({"dim": int(E.shape[1]), "seconds": round(secs, 2), "rows": int(E.shape[0]),
                   "sparse": bool(sp.issparse(E))}, open(f_meta, "w"))
    return l2norm(E) if normalize else E


def embedding_meta(tag, name) -> dict:
    f = C.CACHE_DIR / tag / f"{name}.json"
    return json.load(open(f)) if f.exists() else {}
