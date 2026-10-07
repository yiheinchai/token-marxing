# Lab notebook

## Exp 0 (j001): baseline 3.2466. 121k params, ~1517 steps, ~104k tok/s, ~2 epochs of train data.
Final train loss ~2.2-2.3 nats is about val (3.25 bits), so the model is undertrained, not overfitting.

## Exp 1: adamw-cosine
Hypothesis: plain SGD (lr 1e-2, mom 0.9, constant) is the bottleneck. AdamW at a 3e-3 peak with time-based
3% warmup and cosine decay to 5% should converge much further in 60s. Same model, wd=0, betas (0.9, 0.99).
Next: if it wins, keep the optimizer and scale capacity (CONTEXT 16, HIDDEN 512-1024, EMBED 32, GELU/2 layers).
Check that tok/s stays near 100k. If it loses, try lr 1e-3 or 1e-2.

## Exp 2: ctx16-h384
Exp1 (AdamW+cosine) 2.4437 kept. Hypothesis: we are compute-bound (~75 GFLOP/s single thread) and
~6M tokens for 121k params is far past compute-optimal, so a ~2x bigger model with more context
should win. CONTEXT 8->16, HIDDEN 256->384 (EMBED 24) -> ~252k params, expect ~50k tok/s, ~3M tokens.
Next: if it wins, try GELU + a 2nd hidden layer or LR 5e-3; if tok/s drops too far, back off HIDDEN.

## Exp 3: offset-tables
Exp2 2.5042 discard: 60k tok/s (vs 91k), only 874 steps, so extra FLOPs cost more than they gave. We are compute-bound.
Hypothesis: emb+fc1 is linear, so it equals a sum of per-offset [256,HIDDEN] lookup tables. Using EmbeddingBag(sum) there is
full-rank (a superset) and drops the 192x256 matmul (~40% of FLOPs), giving more tokens AND more capacity. CONTEXT 8 / HIDDEN 256 / tanh kept; ~590k params.
Next: if it wins, spend the freed compute on CONTEXT 16 (almost free now) or a residual 2nd layer / GELU; if tok/s barely rises, check Adam cost on the tables.

## Exp 4 (final): adamw-lr8e-3
Exp3 2.5699 discard. Every change that adds params or FLOPs has lost because we are compute-bound, so the best remaining lever costs nothing per step: optimisation.
Hypothesis: with only ~1500 steps and an undertrained model, a 3e-3 peak is too cautious. Raising peak LR to 8e-3 (warmup 3%->5% for stability, cosine to 5% kept) should cut loss further at identical tok/s. Arch is unchanged from exp1.
Next (if there were more runs): sweep LR 5e-3/1.5e-2, then try BATCH_SIZE 16 (more steps) or ReLU (cheaper than tanh).
