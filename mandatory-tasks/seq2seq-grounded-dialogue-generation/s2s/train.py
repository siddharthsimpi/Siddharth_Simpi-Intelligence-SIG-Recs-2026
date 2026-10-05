"""Training utilities: Adam, gradient clipping, length-bucketed batching, training loop with early stopping."""
from __future__ import annotations

import json
import time

import numpy as np

from .tokenize import pad_batch


class Adam:
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, betas[0], betas[1], eps, 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m *= self.b1; m += (1 - self.b1) * p.grad
            v *= self.b2; v += (1 - self.b2) * p.grad * p.grad
            p.data -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def clip_grad_norm(params, max_norm):
    total = float(np.sqrt(sum(float((p.grad ** 2).sum()) for p in params if p.grad is not None)))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad *= scale
    return total


def collate(examples, pad, bos, eos, n_sources):
    """examples: list of {'srcs': [ids per source], 'tgt': ids}. -> args for Seq2Seq.loss."""
    srcs = []
    for k in range(n_sources):
        ids, _, mask = pad_batch([e["srcs"][k] for e in examples], pad)
        srcs.append((ids, mask))
    tin, _, tmask = pad_batch([[bos] + e["tgt"] for e in examples], pad)
    tout, _, _ = pad_batch([e["tgt"] + [eos] for e in examples], pad)
    return srcs, tin, tout, tmask


def iterate_batches(examples, batch_size, rng=None, chunk=50):
    """Yield lists of examples. With rng: shuffle, then sort inside chunks of `chunk` batches by length
    (less padding => faster) and shuffle the resulting batches."""
    idx = np.arange(len(examples))
    if rng is not None:
        rng.shuffle(idx)
    key = lambda i: (len(examples[i]["tgt"]), sum(len(s) for s in examples[i]["srcs"]))
    batches = []
    span = batch_size * chunk
    for s in range(0, len(idx), span):
        part = sorted(idx[s:s + span], key=key) if rng is not None else list(idx[s:s + span])
        batches += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    if rng is not None:
        rng.shuffle(batches)
    for b in batches:
        yield [examples[i] for i in b]


def eval_loss(model, examples, collate_fn, batch_size=64):
    """Mean per-token negative log-likelihood and perplexity (no dropout)."""
    nll = n = 0.0
    from . import autograd as ag
    with ag.no_grad():
        for b in iterate_batches(examples, batch_size, None):
            loss = model.loss(*collate_fn(b), training=False)
            nll += loss.aux[0]
            n += loss.aux[1]
    mean = nll / max(n, 1)
    return mean, float(np.exp(min(mean, 50)))


def train(model, train_ex, val_ex, collate_fn, epochs=10, batch_size=32, lr=2e-3, clip=5.0, patience=3,
          lr_decay=0.5, seed=0, log=print, max_seconds=None, max_steps_per_epoch=None, ckpt_path=None, tag=""):
    """Teacher-forced training with Adam, gradient clipping, LR decay on plateau and early stopping on validation
    loss. The best-validation weights are restored at the end. Returns the history dict."""
    params = model.parameters()
    opt = Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_ppl": [], "lr": [], "seconds": []}
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    for ep in range(1, epochs + 1):
        tot = cnt = 0.0
        te = time.time()
        for step, b in enumerate(iterate_batches(train_ex, batch_size, rng)):
            if max_steps_per_epoch and step >= max_steps_per_epoch:
                break
            model.zero_grad()
            loss = model.loss(*collate_fn(b), training=True)
            loss.backward()
            clip_grad_norm(params, clip)
            opt.step()
            tot += loss.aux[0]; cnt += loss.aux[1]
            if max_seconds and time.time() - t0 > max_seconds:
                break
        tr = tot / max(cnt, 1)
        vl, vp = eval_loss(model, val_ex, collate_fn, batch_size)
        for k, v in zip(hist, (ep, tr, vl, vp, opt.lr, time.time() - te)):
            hist[k].append(v)
        log(f"[{tag}] epoch {ep:2d}  train nll {tr:.4f}  val nll {vl:.4f}  val ppl {vp:.2f}  lr {opt.lr:.5f}  ({time.time() - te:.0f}s)")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.copy() for k, v in model.state_dict().items()}
            if ckpt_path:
                model.save(ckpt_path)
        else:
            bad += 1
            opt.lr *= lr_decay
            if bad >= patience:
                log(f"[{tag}] early stopping (no improvement for {patience} epochs)")
                break
        if max_seconds and time.time() - t0 > max_seconds:
            log(f"[{tag}] stopping: time budget of {max_seconds}s reached")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val_loss"] = best
    hist["train_seconds"] = time.time() - t0
    return hist


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
