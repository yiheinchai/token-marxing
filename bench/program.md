# Autoresearch: minimise val_bpb

You are an autonomous ML research lab. Goal: the lowest `val_bpb` (validation bits per byte,
lower is better) on a byte-level language-modelling task, within a fixed number of experiments.

## Files
- `prepare.py`: fixed data/eval code and constants (time budget, param cap). Read-only.
- `train.py`: model + training loop. **The only file whose code may change.** Keep its contract
  (see its docstring): `build_model()`, causal, <= MAX_PARAMS params, prints `val_bpb:`.
- `check.py`: end-to-end pre-submission check (contract + 3s smoke run). Must print `CHECK PASSED`
  before a job is submitted.
- `jobq.py`: the cluster queue. `python jobq.py --help`. Jobs sit PENDING in the queue for a
  while, then train for 60s on one CPU thread. There is no blocking wait: poll `status`.
  `NODE_FAIL` is an infrastructure failure: resubmit the same code unchanged.
  `FAILED` means train.py crashed: read `python jobq.py logs <id>`.
- `results.tsv`: experiment log (tab-separated, header already present).

## Protocol
Experiment 0 is the unmodified baseline: check, submit, wait, record it, `keep`.
Then run exactly 4 further experiments (1..4). For each:
1. Change `train.py` to test ONE idea.
2. `python check.py` -> must pass; fix and re-check if it fails.
3. `python jobq.py submit --name <short-name>`, then poll until the job finishes.
   Tip: a job takes ~1.5-2 min including queueing; wait with a single shell polling loop
   (e.g. `until python jobq.py status ID | grep -qE 'COMPLETED|FAILED|NODE_FAIL'; do sleep 15; done`)
   rather than many separate status calls.
4. `python jobq.py result <id>` for the metrics.
5. Append a row to `results.tsv`: `exp<TAB>job<TAB>val_bpb<TAB>status<TAB>description`
   where status is `keep` (beat the best so far), `discard` or `crash`.
6. keep -> `git commit -am "exp<N>: <description> val_bpb=<x>"`; discard/crash -> `git checkout train.py`.

A crash or a check that cannot be fixed still counts as one experiment. NODE_FAIL resubmits do not.
When experiment 4 is recorded, stop and print a short summary: best val_bpb and what worked.
Work autonomously; never ask questions.
