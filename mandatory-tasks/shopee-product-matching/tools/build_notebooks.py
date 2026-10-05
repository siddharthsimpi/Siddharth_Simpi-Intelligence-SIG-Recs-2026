"""Convert percent-format scripts (PartX/notebook.py) into Jupyter notebooks (PartX/notebook.ipynb).

    python tools/build_notebooks.py
Cells start with `# %%` (code) or `# %% [markdown]` (markdown; lines are `# text`).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["PartA", "PartB", "PartC", "Finale"]


def parse(text):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        body = "\n".join(buf).strip("\n")
        if kind == "markdown":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
        if body.strip():
            src = body.split("\n")
            src = [l + "\n" for l in src[:-1]] + [src[-1]]
            c = {"cell_type": kind, "metadata": {}, "source": src}
            if kind == "code":
                c.update(execution_count=None, outputs=[])
            cells.append(c)

    for line in text.split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def main():
    for part in TARGETS:
        src = ROOT / part / "notebook.py"
        if not src.exists():
            continue
        nb = {"cells": parse(src.read_text(encoding="utf-8")),
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}},
              "nbformat": 4, "nbformat_minor": 4}
        (ROOT / part / "notebook.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"built {part}/notebook.ipynb ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
