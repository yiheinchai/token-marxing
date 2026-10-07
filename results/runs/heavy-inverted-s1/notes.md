# Lab notebook

- exp1 (adamw-cosine): Baseline 3.417 bpb looks heavily under-optimised (SGD+mom lr=1e-2, constant LR), not capacity-limited,
  so the optimiser is the bottleneck. Swapped to AdamW lr=3e-3 (betas 0.9/0.99, wd=0), 2% time-based warmup, then cosine to 5% of peak; arch unchanged (121k params).
  Next: if it helps, scale capacity (CONTEXT 8->16, EMBED 24->48, HIDDEN 256->1024) while keeping an eye on tok/s on one CPU thread; maybe batch 32->64.
- exp2 (ngram-tables): exp1 = 2.517. Rejected naive 16/48/1024 scaling: ~9x FLOPs/token -> ~9x fewer tokens in 60s, and exp1 (121k params, ~2.4M tokens) is already near compute-optimal. Instead
  fold emb+fc1 into per-offset [256,HIDDEN] lookup tables summed via EmbeddingBag (exactly the same function class, minus the EMBED rank bottleneck), so layer 1 is gather+sum with no matmul.
  CONTEXT 8->16, HIDDEN 256->512, 2.23M params, roughly exp1's FLOPs/token (fc2 dominates); table init std 0.6/sqrt(CONTEXT). LR/schedule unchanged.
  Next: compare tok/s with exp1. If it wins, add a 2nd hidden layer or try a higher LR (1e-2); if it loses, check whether tables are under-updated (try a separate higher LR for the tables).
- exp3 (adamw-lr1e-2): exp2 = 2.864 (discard). WARNING: the working train.py was the SGD baseline, not exp1's AdamW, even though git HEAD has an "exp1" commit. So exp1 was rebuilt from this notebook (AdamW b=0.9/0.99, wd=0, 2% time warmup, cosine to 5%).
  Hypothesis: 121k-param MLP gets only ~600 steps in 60s, so it is under-trained and a 3e-3 peak LR is too timid. Only change vs exp1: peak LR 3e-3 -> 1e-2 (arch, batch, schedule shape the same).
  Next: if it wins, try 2e-2 or a 2nd hidden layer (HIDDEN 256, about +66k params, small FLOP cost). If it is unstable or loses, go back to 3e-3 and try a 2nd layer or BATCH 32->16 for more steps.
- exp4 (adamw-lr2e-2): exp3 = 2.355 (keep). Going from 3e-3 to 1e-2 gained 0.16 bpb, so the LR optimum for about 600 steps is probably still higher. Final experiment, so I continue the LR sweep (the lowest-risk, highest-expected-gain option) rather than add arch changes.
  Only change: peak LR 1e-2 -> 2e-2 (same arch, batch 32, betas 0.9/0.99, 2% warmup, cosine to 5%).
  Next (if there were more budget): if 2e-2 wins, try 3e-2 plus grad clip 1.0; if it loses, keep 1e-2 and try BATCH 32->16 (twice the steps) or a 2nd hidden layer.
