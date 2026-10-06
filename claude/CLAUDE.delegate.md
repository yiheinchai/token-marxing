# Division of labour: you think, the operator does the labour

You run on an expensive model; your tokens are for research thinking and writing PyTorch code.
All mechanical pipeline work is done by the `operator` subagent (cheap model):
running check.py, submitting / polling / resubmitting jobs, reading logs and metrics,
appending to results.tsv and git keep/discard.

- Never run check.py, jobq.py, `sleep`, or read job logs yourself.
- Per experiment: edit train.py, then make ONE operator call:
  "exp <N>, name <short-name>, desc <one line>". Read its <=8-line report.
- If it reports CHECK FAILED or a crash you want to fix, fix train.py and call it again.
- You still own: choosing ideas, editing train.py, and the final summary.
