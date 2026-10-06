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
  This is a busy shared cluster: there is no `result` command; the metrics (`val_bpb:` etc.)
  are printed at the end of each job's log.
- `results.tsv`: experiment log (tab-separated, header already present).

## Protocol
Experiment 0 is the unmodified baseline: check, submit, wait, record it, `keep`.
Then run exactly 4 further experiments (1..4). For each:
1. Change `train.py` to test ONE idea.
2. `python check.py` -> must pass; fix and re-check if it fails.
3. Every experiment is run with TWO seeds, as two jobs:
   `python jobq.py submit --name <short-name>-s1 --seed 1` and the same with `--seed 2`.
   Poll until both jobs finish.
4. Read each job's final metrics from its log. The experiment's val_bpb is the mean of the two.
5. Append a row to `results.tsv`: `exp<TAB>job_ids<TAB>mean_val_bpb<TAB>status<TAB>description`
   where status is `keep` (beat the best so far), `discard` or `crash`.
6. keep -> `git commit -am "exp<N>: <description> val_bpb=<x>"`; discard/crash -> `git checkout train.py`.

A crash or a check that cannot be fixed still counts as one experiment. NODE_FAIL resubmits do not.
When experiment 4 is recorded, stop and print a short summary: best val_bpb and what worked.
Work autonomously; never ask questions.
