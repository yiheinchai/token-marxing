"""Fixed infrastructure for the toy autoresearch task. DO NOT EDIT.

Byte-level language modelling on Python standard-library source code.
The metric is validation bits-per-byte (val_bpb, lower is better), evaluated
on a fixed set of held-out windows so runs are comparable.

train.py imports from here: constants, get_batch(), evaluate_bpb().
"""
import glob
import hashlib
import math
import os
import sysconfig

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

VOCAB_SIZE = 256          # raw bytes
SEQ_LEN = 128             # context length used for evaluation
TIME_BUDGET_S = 60        # wall-clock training budget per run (seconds)
MAX_PARAMS = 3_000_000    # parameter cap
NUM_THREADS = 1           # every job gets exactly one CPU thread
EVAL_WINDOWS = 96         # number of fixed validation windows
TRAIN_BYTES = 3_000_000
VAL_BYTES = 300_000


def _build_corpus():
    os.makedirs(DATA_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    buf = bytearray()
    for f in files:
        with open(f, "rb") as fh:
            buf += fh.read()
        if len(buf) >= TRAIN_BYTES + VAL_BYTES:
            break
    buf = bytes(buf[: TRAIN_BYTES + VAL_BYTES])
    np.frombuffer(buf[:TRAIN_BYTES], dtype=np.uint8).tofile(os.path.join(DATA_DIR, "train.bin"))
    np.frombuffer(buf[TRAIN_BYTES:], dtype=np.uint8).tofile(os.path.join(DATA_DIR, "val.bin"))
    with open(os.path.join(DATA_DIR, "sha256.txt"), "w") as fh:
        fh.write(hashlib.sha256(buf).hexdigest() + "\n")


def _load(split):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    if not os.path.exists(path):
        _build_corpus()
    return torch.from_numpy(np.fromfile(path, dtype=np.uint8).astype(np.int64))


_cache = {}


def data(split):
    if split not in _cache:
        _cache[split] = _load(split)
    return _cache[split]


def get_batch(split, batch_size, seq_len=SEQ_LEN, generator=None):
    """Random (x, y) windows: y is x shifted by one byte."""
    d = data(split)
    ix = torch.randint(0, len(d) - seq_len - 1, (batch_size,), generator=generator)
    x = torch.stack([d[i : i + seq_len] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + seq_len] for i in ix])
    return x, y


@torch.no_grad()
def evaluate_bpb(model):
    """Mean next-byte cross-entropy over fixed validation windows, in bits/byte.

    The model must map LongTensor[B, T] -> logits FloatTensor[B, T, 256] and be causal.
    """
    model.eval()
    d = data("val")
    g = torch.Generator().manual_seed(1234)
    ix = torch.randint(0, len(d) - SEQ_LEN - 1, (EVAL_WINDOWS,), generator=g)
    total, count = 0.0, 0
    for chunk in ix.split(32):
        x = torch.stack([d[i : i + SEQ_LEN] for i in chunk])
        y = torch.stack([d[i + 1 : i + 1 + SEQ_LEN] for i in chunk])
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), reduction="sum")
        total += loss.item()
        count += y.numel()
    model.train()
    return total / count / math.log(2)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    _build_corpus()
    print("train bytes", len(data("train")), "val bytes", len(data("val")))
