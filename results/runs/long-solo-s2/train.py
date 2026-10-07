"""Editable training script for the toy autoresearch task.

Contract (checked by check.py):
  * build_model() returns an nn.Module mapping LongTensor[B, T] -> logits[B, T, 256]
  * the model is causal (logits at position t depend only on bytes <= t)
  * parameter count <= prepare.MAX_PARAMS
  * `python train.py` trains for prepare.TIME_BUDGET_S seconds and prints `val_bpb: <float>`
Everything else (architecture, optimiser, schedule, batch size, ...) is fair game.
"""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import prepare

# ---------------------------------------------------------------- hyperparams
CONTEXT = 8          # bytes of left context the MLP sees
EMBED = 24
HIDDEN = 256
BATCH_SIZE = 32
LR = 6e-3
LOG_EVERY = 25


class CausalMLP(nn.Module):
    """Bengio-style n-gram MLP: concat embeddings of the previous CONTEXT bytes."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(prepare.VOCAB_SIZE, EMBED)
        self.fc1 = nn.Linear(CONTEXT * EMBED, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, prepare.VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.shape
        padded = F.pad(idx, (CONTEXT - 1, 0), value=0)        # left pad -> causal
        windows = padded.unfold(1, CONTEXT, 1)                 # [B, T, CONTEXT]
        h = self.emb(windows).reshape(B, T, CONTEXT * EMBED)
        return self.fc2(torch.tanh(self.fc1(h)))


def build_model():
    return CausalMLP()


def main():
    torch.set_num_threads(prepare.NUM_THREADS)
    torch.manual_seed(int(os.environ.get("SEED", 0)))
    budget = 3.0 if os.environ.get("SMOKE") else prepare.TIME_BUDGET_S
    model = build_model()
    n_params = prepare.count_params(model)
    print(f"params: {n_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.01)

    t0 = time.time()
    step, tokens = 0, 0
    while time.time() - t0 < budget:
        x, y = prepare.get_batch("train", BATCH_SIZE)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, prepare.VOCAB_SIZE), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        tokens += x.numel()
        if step % LOG_EVERY == 0:
            el = time.time() - t0
            gnorm = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            print(f"step {step:5d} | loss {loss.item():.4f} | grad_norm {gnorm:.3f} | "
                  f"lr {opt.param_groups[0]['lr']:.2e} | tok/s {tokens / el:,.0f} | elapsed {el:5.1f}s",
                  flush=True)

    train_time = time.time() - t0
    val_bpb = prepare.evaluate_bpb(model)
    print(f"steps: {step}")
    print(f"train_time_s: {train_time:.1f}")
    print(f"val_bpb: {val_bpb:.4f}")


if __name__ == "__main__":
    main()
