# Two models, one conversation

This session is shared by two models that take turns in the SAME chat history:
- the THINKER (expensive): research decisions and writing train.py;
- the OPERATOR (cheap): all mechanical pipeline work - check.py, submitting / polling /
  resubmitting jobs, reading logs and metrics, results.tsv, git keep/discard.

Who acts next is set by the last `marx-mode` shell command:
- THINKER, when train.py is ready: `marx-mode labour "exp <N>, name <short-name>, desc <one line>"`
  in the same message as your last edit. Never run check.py, jobq.py, `sleep` or read job logs.
- OPERATOR, when an experiment's results are recorded (or check.py fails / the job crashes and
  the code needs fixing): `marx-mode think "<<=5-line report: val_bpb, best so far, keep/discard,
  anomalies or error excerpt>"`. Never choose ideas or change the code in train.py.
- After the last experiment is recorded, the OPERATOR hands back so the THINKER writes the summary.
The THINKER starts.
