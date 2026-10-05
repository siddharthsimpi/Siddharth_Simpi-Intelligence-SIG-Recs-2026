"""Hyper-parameter grids shared by Task 2 and Task 3 (every family gets a comparable, small tuning budget).

All grids were chosen BEFORE looking at the test set; selection uses validation log loss only.
"""
from __future__ import annotations


def spec_name(s: dict) -> str:
    t = s["type"]
    extra = ""
    if s.get("dropout"):
        extra += f" do{s['dropout']}"
    if s.get("weight_decay"):
        extra += f" wd{s['weight_decay']:g}"
    if s.get("emb_init"):
        extra += f" init{s['emb_init']}"
    lr = f" lr{s.get('lr', 1e-3):g}"
    if t == "logreg":
        return f"LogReg{lr}"
    if t == "fm":
        return f"FM e{s['emb_dim']}{lr}"
    if t == "mlp":
        return f"MLP e{s['emb_dim']} {list(s['hidden'])}{extra}"
    if t == "dcn":
        return f"DCN e{s['emb_dim']} x{s['n_cross']} {list(s['hidden'])}{extra}"
    if t == "dlrm":
        inter = "" if s.get("interaction", "dot") == "dot" else " NO-INTERACTION(cat)"
        return f"DLRM e{s['emb_dim']} bot{list(s['bottom'])} top{list(s['top'])}{extra}{inter}"
    return t


LR = 3e-4

LOGREG_GRID = [{"type": "logreg", "lr": 3e-3}]
FM_GRID = [{"type": "fm", "emb_dim": 4, "lr": LR}, {"type": "fm", "emb_dim": 8, "lr": LR}]

MLP_GRID = [
    {"type": "mlp", "emb_dim": 4, "hidden": [64], "dropout": 0.1, "lr": LR},
    {"type": "mlp", "emb_dim": 8, "hidden": [128, 64], "dropout": 0.0, "lr": LR},                       # no regularisation
    {"type": "mlp", "emb_dim": 8, "hidden": [128, 64], "dropout": 0.2, "lr": LR},
    {"type": "mlp", "emb_dim": 8, "hidden": [128, 64], "dropout": 0.2, "weight_decay": 1e-3, "lr": LR},
    {"type": "mlp", "emb_dim": 8, "hidden": [128, 64], "dropout": 0.3, "weight_decay": 1e-2, "lr": LR},
    {"type": "mlp", "emb_dim": 16, "hidden": [128, 64], "dropout": 0.3, "weight_decay": 1e-3, "lr": LR},
    {"type": "mlp", "emb_dim": 16, "hidden": [256, 128, 64], "dropout": 0.3, "lr": LR},
    {"type": "mlp", "emb_dim": 8, "hidden": [256, 128, 64], "dropout": 0.3, "weight_decay": 1e-3, "lr": LR},
    {"type": "mlp", "emb_dim": 16, "hidden": [512, 256, 128], "dropout": 0.0, "lr": LR},                # big, unregularised
]

DCN_GRID = [
    {"type": "dcn", "emb_dim": 8, "hidden": [128, 64], "n_cross": 2, "dropout": 0.2, "lr": LR},
    {"type": "dcn", "emb_dim": 8, "hidden": [128, 64], "n_cross": 3, "dropout": 0.2, "lr": LR},
    {"type": "dcn", "emb_dim": 8, "hidden": [128, 64], "n_cross": 3, "dropout": 0.2, "weight_decay": 1e-3, "lr": LR},
    {"type": "dcn", "emb_dim": 8, "hidden": [128, 64], "n_cross": 3, "dropout": 0.3, "weight_decay": 1e-2, "lr": LR},
    {"type": "dcn", "emb_dim": 16, "hidden": [256, 128, 64], "n_cross": 4, "dropout": 0.3, "lr": LR},
]

DLRM_GRID = [
    {"type": "dlrm", "emb_dim": 4, "bottom": [16], "top": [64, 32], "dropout": 0.2, "lr": LR},
    {"type": "dlrm", "emb_dim": 8, "bottom": [32], "top": [128, 64], "dropout": 0.2, "lr": LR},
    {"type": "dlrm", "emb_dim": 8, "bottom": [32], "top": [128, 64], "dropout": 0.3, "weight_decay": 1e-3, "lr": LR},
    {"type": "dlrm", "emb_dim": 8, "bottom": [32], "top": [128, 64], "dropout": 0.3, "weight_decay": 1e-2, "lr": LR},
    {"type": "dlrm", "emb_dim": 4, "bottom": [16], "top": [32], "dropout": 0.3, "weight_decay": 1e-3, "lr": LR},
    {"type": "dlrm", "emb_dim": 16, "bottom": [64], "top": [128, 64], "dropout": 0.3, "weight_decay": 1e-2, "lr": LR},
]


def quick(grid):
    return grid[:2]
