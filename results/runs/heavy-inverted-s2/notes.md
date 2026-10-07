# Lab notebook

## exp1 adamw-sched
Hypothesis: the baseline (3.2904) is limited by optimisation, not capacity. Plain SGD at lr 1e-2 with no schedule under-trains in 60s.
Change: SGD -> AdamW (lr 3e-3, betas 0.9/0.99, wd 0), time-based schedule: 3% linear warmup, then linear decay to 10% of peak. Architecture unchanged (8 ctx, 24 emb, 256 hidden).
Next: if this wins, spend capacity well under the 3M cap, e.g. a longer context (16-32) plus a wider hidden layer (512-1024) with GELU, then try the LR peak (2e-3 vs 6e-3) or batch size. If it loses, sweep the LR down.
Result: 2.4506 (keep). Logs: ~85k tok/s, ~1.25k steps (~5M tok, ~1.7 epochs), train loss still falling at end (~1.5 nats vs val 1.70 nats).

## exp2 ctx16-gelu-h384
Hypothesis: 8 bytes of context is too short for code (indentation level, identifiers), and the 121k-param model is limited by capacity. A longer window plus a moderate widening should beat the ~1.7x throughput loss.
Change: CONTEXT 8->16, EMBED 24->16 (fc1 input 192->256), HIDDEN 256->384, tanh->GELU. ~201k params, ~197k MAC/token (was ~115k). Optimiser and schedule unchanged.
Next: if it wins, raise the peak LR (6e-3) or try ctx 32. If it loses because of throughput, go back to hidden 256 and keep ctx 16 to isolate the context effect.
Result: 2.5187 (discard).

## exp3 ctx16-emb12
NOTE: before this experiment, train.py was the plain SGD baseline, not exp1. The exp2 discard was committed (see git log) instead of being reverted to exp1. I rebuilt exp1 from the description above: AdamW 3e-3, betas 0.9/0.99, wd 0, 3% warmup, linear decay to 10%.
Hypothesis: exp2 changed four things at once and lost about 1.7x throughput. A longer context at the same compute should help on code.
Change: on top of exp1, CONTEXT 8->16 and EMBED 24->12, so fc1 fan-in stays 192 and MAC/token and throughput match exp1. Params ~118k. Hidden 256 and tanh unchanged.
Next: if it wins, try ctx 32 with emb 6-8, or raise the peak LR (6e-3 to 1e-2) since only ~1.25k steps fit. If it loses, context is not the bottleneck: go back to ctx8/emb24 and sweep the LR (1e-2).
Result: 2.6002 (discard).

## exp4 adamw-lr1e-2
NOTE: train.py was again the plain SGD baseline (HEAD does not hold exp1 code). I rebuilt exp1 exactly: ctx8/emb24/h256/tanh, AdamW betas 0.9/0.99, wd 0, 3% warmup, linear decay to 10%.
Hypothesis: two longer-context runs lost, so the bottleneck is optimisation in ~1.25k steps (train loss still falling at the end of exp1), not context. A higher peak LR should get further in the same steps.
Change: only the peak LR, 3e-3 -> 1e-2. Architecture, batch and schedule shape are the same as exp1.
Next: if it wins, try 2e-2 or batch 16 for more steps. If it diverges or loses, try 6e-3, or init the fc2 bias to log unigram frequencies.
