"""Matplotlib helpers (headless). Picks a Devanagari-capable font when one is installed."""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

_FONT_CANDIDATES = ["Nirmala UI", "Mangal", "Noto Sans Devanagari", "Lohit Devanagari", "Kohinoor Devanagari",
                    "Devanagari Sangam MN", "Arial Unicode MS", "DejaVu Sans"]


def setup_fonts():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _FONT_CANDIDATES[:-1]:
        if name in available:
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return name
    # no Devanagari font installed: plots still work, Hindi tick labels may show as empty boxes
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    return None


setup_fonts()


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_histories(histories: dict, path, key="val_loss", title="Validation loss (nats / token)"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in histories.items():
        ax[0].plot(h["epoch"], h["train_loss"], marker="o", label=name)
        ax[1].plot(h["epoch"], h["val_loss"], marker="o", label=name)
    ax[0].set(title="Training loss (nats / token, with dropout)", xlabel="epoch")
    ax[1].set(title=title, xlabel="epoch")
    ax[0].legend(fontsize=8)
    _save(fig, path)


def plot_bleu_by_length(tables: dict, path, ylabel="BLEU"):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, rows in tables.items():
        ax.plot([r["bucket"] for r in rows], [r["bleu"] for r in rows], marker="o", label=name)
    ax.set(title=f"{ylabel} by source length (test)", xlabel="source length (tokens)", ylabel=ylabel)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_bars(labels, values, path, title, ylabel, errs=None):
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(labels)), 4))
    ax.bar(range(len(labels)), values, yerr=errs, color="#4C72B0", capsize=3)
    ax.set(title=title, ylabel=ylabel, xticks=range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    _save(fig, path)


def plot_attention(alpha, src_tokens, out_tokens, path, title="Attention weights"):
    """alpha: (T_out, S) matrix."""
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(src_tokens) + 2), max(3, 0.4 * len(out_tokens) + 1.5)))
    im = ax.imshow(alpha, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(len(out_tokens)))
    ax.set_yticklabels(out_tokens, fontsize=8)
    ax.set(title=title, xlabel="source tokens", ylabel="generated tokens")
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, path)
