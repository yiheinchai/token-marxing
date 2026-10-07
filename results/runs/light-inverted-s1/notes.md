# Lab notebook

- exp1 (adam-lr2e-3): Baseline 3.3845 bpb is far too high for byte-level code, so the model is badly
  undertrained. SGD at an effective lr of 1e-2 (LR*10) is the likely bottleneck. Changed the optimizer to Adam with lr=2e-3; nothing else changed.
  I set aside the operator's CONTEXT=16 suggestion because more context won't help a model that hasn't learned to use the context it has.
  Next: if this wins, add warmup plus cosine/linear decay to 0, then a larger batch or wider HIDDEN, then CONTEXT=16.
- exp2 (adam-warmup-cosine): Result: exp1 won, 3.3845 -> 2.4807. Hypothesis: a constant LR leaves the end-of-run noise floor high, so annealing to 0 should give a cheap gain
  in a fixed 60s budget. Peak LR stays 2e-3 so the schedule is the only change. Changed: a time-based schedule (elapsed/budget, since the step count is unknown) with 3% linear warmup, then cosine decay to 0.
  Next: if it wins, raise the peak LR (5e-3) under the schedule, then try wider HIDDEN/EMBED or CONTEXT=16 (the param budget is mostly unused at ~121k of 3M).
- exp3 (adam-lr5e-3): Result: exp2 lost badly, 2.4807 -> 2.7171. Cosine roughly halves the integrated LR, and that cost 0.24 bpb, so the run is
  limited by optimisation progress, not by end-of-run noise. Hypothesis: a higher constant LR moves further in 60s. Changed: constant Adam LR 2e-3 -> 5e-3, nothing else (the schedule was reverted).
  I deferred the operator's capacity/context ideas because a bigger model takes fewer steps and can't help while optimisation is the bottleneck.
  Next: if it wins, try LR 1e-2 or a larger HIDDEN (more capacity at a similar step cost). If it loses, LR ~2e-3 is near-optimal, so move on to HIDDEN=512 / CONTEXT=16.
- exp4 (adam-lr1e-2): Result: exp3 won, 2.4807 -> 2.4164. Every result so far says the run is optimisation-limited. My rough estimate is only ~700 steps
  of 4096 tokens in 60s, so a wider net or more context would cut the step count further. Hypothesis: the LR optimum is still above 5e-3. Changed: constant Adam LR 5e-3 -> 1e-2,
  nothing else. I again deferred the operator's CONTEXT=16/wider ideas for the same reason. Next (if there were budget): if it loses, keep 5e-3 and try BATCH_SIZE=16
  for 2x steps; if it wins, try 2e-2 plus grad clipping, then a residual/LayerNorm hidden layer to fix tanh saturation at high LR.
