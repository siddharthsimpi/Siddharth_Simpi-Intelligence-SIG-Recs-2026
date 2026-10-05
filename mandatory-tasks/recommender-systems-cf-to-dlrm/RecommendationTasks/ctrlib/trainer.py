"""Mini-batch Adam training with early stopping on validation log loss, plus experiment helpers."""
from __future__ import annotations

import time

import numpy as np

from . import nn
from .metrics import all_metrics, best_f1_threshold, log_loss, roc_auc
from .models import Schema, build_model


def predict_proba(model, data, batch=4096):
    dense, cat, _ = data
    out = [model.forward(dense[i:i + batch], cat[i:i + batch], training=False) for i in range(0, len(cat), batch)]
    return nn.sigmoid(np.concatenate(out))


def fit(model, train, val, lr=1e-3, batch_size=512, epochs=20, patience=3, seed=0, verbose=False):
    """Returns history dict. Restores the weights of the epoch with the best validation log loss."""
    params = model.params()
    opt = nn.Adam(params, lr=lr)
    dense, cat, y = train
    rng = np.random.default_rng(seed + 12345)
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": [], "epoch_time": []}
    best, best_state, bad, best_epoch = np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = rng.permutation(len(y))
        for s in range(0, len(y), batch_size):
            b = perm[s:s + batch_size]
            opt.zero_grad()
            logits = model.forward(dense[b], cat[b], training=True)
            _, g = nn.bce_with_logits(logits, y[b])
            model.backward(g)
            opt.step()
        epoch_time = time.time() - t0
        ptr, pva = predict_proba(model, train), predict_proba(model, val)
        tl, vl, va = log_loss(y, ptr), log_loss(val[2], pva), roc_auc(val[2], pva)
        for k, v in zip(hist, (epoch, tl, vl, va, epoch_time)):
            hist[k].append(v)
        if verbose:
            print(f"    epoch {epoch:2d}  train {tl:.4f}  val {vl:.4f}  val-auc {va:.4f}  ({epoch_time:.1f}s)")
        if vl < best - 1e-5:
            best, bad, best_epoch = vl, 0, epoch
            best_state = [p.value.copy() for p in params]
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        for p, v in zip(params, best_state):
            p.value[...] = v
    hist["best_epoch"] = best_epoch
    return hist


def run_one(spec, seed, data, verbose=False, epochs=20, patience=3):
    """Train one model spec with one seed. Threshold is chosen on validation and frozen for test."""
    schema = Schema(data["pre"].n_dense, data["pre"].cardinalities)
    model = build_model(spec, schema, seed)
    t0 = time.time()
    hist = fit(model, data["train"], data["val"], lr=spec.get("lr", 1e-3), batch_size=spec.get("batch_size", 512),
               epochs=epochs, patience=patience, seed=seed, verbose=verbose)
    train_time = time.time() - t0
    pva, pte = predict_proba(model, data["val"]), predict_proba(model, data["test"])
    thr = best_f1_threshold(data["val"][2], pva)
    t0 = time.time(); predict_proba(model, data["test"]); infer = time.time() - t0
    n = model.n_params()
    return {
        "spec": spec, "seed": seed, "history": hist, "p_val": pva, "p_test": pte, "threshold": thr,
        "val": all_metrics(data["val"][2], pva, thr), "test": all_metrics(data["test"][2], pte, thr),
        "n_params": n, **model.param_breakdown(), "memory_mb_fp32": n * 4 / 1e6,
        "train_time_s": train_time, "sec_per_epoch": float(np.mean(hist["epoch_time"])),
        "infer_us_per_row": infer / len(data["test"][2]) * 1e6, "model": model,
    }
