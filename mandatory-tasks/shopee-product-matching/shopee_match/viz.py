"""Plot helpers."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .embeddings import load_rgb


def textwrap(s, w=28, maxlines=3):
    s = str(s)
    lines = [s[a:a + w] for a in range(0, len(s), w)][:maxlines]
    return "\n".join(lines) + ("…" if len(s) > w * maxlines else "")


def image_grid(df, ids, captions=None, ncols=5, size=2.4, border=None, title=None, save=None):
    """Show listing images + titles. `border` = optional list of colours ('green'/'red'...)."""
    ids = list(ids)
    nrows = int(np.ceil(len(ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * (size + 0.9)))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, i in enumerate(ids):
        ax = axes[k]
        ax.imshow(load_rgb(df.path.iloc[i]))
        cap = captions[k] if captions is not None else df.title.iloc[i]
        ax.set_title(textwrap(cap), fontsize=7)
        if border is not None and border[k]:
            for sp_ in ax.spines.values():
                sp_.set_edgecolor(border[k]); sp_.set_linewidth(3); sp_.set_visible(True)
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig


def pair_figure(df, pairs, captions, save=None, title=None, size=2.6):
    """Each row of `pairs` = (i, j): left image | right image, with a caption per pair."""
    fig, axes = plt.subplots(len(pairs), 2, figsize=(2 * size + 1.5, len(pairs) * (size + 0.4)))
    axes = np.atleast_2d(axes)
    for r, (i, j) in enumerate(pairs):
        for c, g in enumerate((i, j)):
            axes[r, c].imshow(load_rgb(df.path.iloc[g])); axes[r, c].axis("off")
            axes[r, c].set_title(textwrap(df.title.iloc[g], 34), fontsize=7)
        axes[r, 0].text(-0.05, 0.5, textwrap(captions[r], 22, 6), transform=axes[r, 0].transAxes,
                        ha="right", va="center", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig


def hist_compare(a, b, labels, xlabel, title, bins=40, save=None, density=True):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(a, bins=bins, alpha=0.6, label=labels[0], density=density)
    ax.hist(b, bins=bins, alpha=0.6, label=labels[1], density=density)
    ax.set_xlabel(xlabel); ax.set_title(title); ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)
    return fig


def md_table(df, floatfmt="{:.4f}"):
    """DataFrame -> GitHub markdown table (no extra dependency)."""
    def f(x, c):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return ""
            return str(int(x)) if c in ("dim", "n") else floatfmt.format(x)
        return str(x)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(f(x, c) for x, c in zip(r, df.columns)) + " |" for r in df.values]
    return "\n".join([head, sep] + body)
