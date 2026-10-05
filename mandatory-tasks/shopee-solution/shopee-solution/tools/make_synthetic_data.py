"""Generate a tiny Shopee-like dataset (train.csv + train_images/) for SMOKE-TESTING the pipeline only.

    python tools/make_synthetic_data.py --out data/synthetic --groups 300

Products are coloured shapes; listings of one product share colours/shape (with noise, shifts, different
backgrounds) and have noisy titles. This is NOT a substitute for the real Kaggle data.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.fft import dct

WORDS = "bag tas wanita pria sepatu original murah promo baju kaos set cream serum face wash botol 500ml 250ml 100gr hp case iphone samsung tws earbuds wireless baby diaper pants".split()


def phash(img):
    a = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float64)
    d = dct(dct(a, axis=0), axis=1)[:8, :8]
    bits = (d > np.median(d[1:].ravel())).astype(np.uint8).ravel()
    return format(int("".join(map(str, bits)), 2), "016x")


def render(rng, color, shape, bg):
    im = Image.new("RGB", (200, 200), bg)
    dr = ImageDraw.Draw(im)
    x, y, r = rng.randint(60, 140), rng.randint(60, 140), rng.randint(35, 55)
    col = tuple(int(np.clip(c + rng.randint(-15, 15), 0, 255)) for c in color)
    if shape == 0:
        dr.ellipse([x - r, y - r, x + r, y + r], fill=col)
    elif shape == 1:
        dr.rectangle([x - r, y - r, x + r, y + r], fill=col)
    else:
        dr.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=col)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--groups", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    (out / "train_images").mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    # a few "variant" pairs: same title family but different colour => hard negatives
    for g in range(a.groups):
        color = tuple(rng.randint(30, 225) for _ in range(3))
        shape = rng.randint(0, 2)
        base = rng.sample(WORDS, 4)
        for _ in range(rng.choice([1, 2, 2, 3, 3, 4, 5])):
            bg = rng.choice([(255, 255, 255), (240, 240, 240), (250, 235, 215)])
            im = render(rng, color, shape, bg)
            title = base[:]
            if rng.random() < 0.4:
                title.append(rng.choice(WORDS))
            if rng.random() < 0.3:
                title = title[1:]
            rng.shuffle(title)
            t = " ".join(title)
            t = t.upper() if rng.random() < 0.2 else t
            fn = f"{n:08d}.jpg"
            im.save(out / "train_images" / fn)
            rows.append(dict(posting_id=f"train_{n}", image=fn, image_phash=phash(im), title=t, label_group=g))
            n += 1
    pd.DataFrame(rows).sample(frac=1, random_state=0).to_csv(out / "train.csv", index=False)
    print(f"wrote {n} listings / {a.groups} groups -> {out}")


if __name__ == "__main__":
    main()
