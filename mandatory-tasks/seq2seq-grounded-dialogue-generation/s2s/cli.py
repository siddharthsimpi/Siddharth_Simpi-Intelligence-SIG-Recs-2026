"""Command-line options shared by sub-tasks 1 and 2."""
from __future__ import annotations

import os


def add_translation_args(ap, here):
    ap.add_argument("--data", default=None, help="folder or file with the parallel corpus (default: data/translation, "
                                                 "auto-download via gdown / Tatoeba if empty)")
    ap.add_argument("--data-dir", default=os.path.join(here, "..", "data"))
    ap.add_argument("--toy", action="store_true", help="use a tiny SYNTHETIC corpus (smoke test only, not real results)")
    ap.add_argument("--src-lang", default=None, help="source language code (default: inferred from file names, else en)")
    ap.add_argument("--tgt-lang", default=None, help="target language code, e.g. fr or hi (default: inferred from file names, else hi)")
    ap.add_argument("--src-col", default=None)
    ap.add_argument("--tgt-col", default=None)
    ap.add_argument("--max-len", type=int, default=25, help="drop sentence pairs longer than this many tokens")
    ap.add_argument("--max-vocab", type=int, default=6000)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=20000, help="cap on training pairs (CPU time)")
    ap.add_argument("--max-eval", type=int, default=1000, help="cap on validation/test pairs")
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=1500, help="time budget per trained model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny settings for a ~1-2 minute smoke run")
    return ap


def apply_quick(args):
    if args.quick:
        args.epochs, args.max_train, args.max_eval, args.hid, args.emb_dim = 3, 2000, 100, 64, 48
        args.batch_size = 32
    return args


def data_fingerprint(args):
    keys = ["data", "toy", "src_lang", "tgt_lang", "max_len", "max_vocab", "min_freq", "max_train", "max_eval", "seed",
            "emb_dim", "hid", "dropout", "epochs", "lr", "batch_size"]
    return {k: getattr(args, k) for k in keys}
