# token-marxing

> *From each model according to its price, to each task according to its difficulty.*

When Claude Code runs an autoresearch loop, Opus spends most of its money on **labour**:
running `.py` files, e2e checks, queueing jobs, waiting for jobs, reading logs, `git`.
token-marxing gives that labour to a cheap model (DeepSeek V4.1 Flash, Haiku, ...) and
keeps Opus for the **intellectual work**: deciding what to try and writing the PyTorch code.

It has two parts:

1. **A division of labour inside Claude Code**: Opus stays the main agent and hands every
   "check, submit, wait, record" cycle to an [`operator`](claude/agents/operator.md) subagent
   on the cheap slot, which reports back in ≤8 lines. The repo also has the inverted form, where
   the cheap model drives and calls an Opus [`researcher`](claude/agents/researcher.md). It works,
   but it is not recommended (see results).
2. **[`marx/router.py`](marx/router.py)**: a stdlib-only proxy you point `ANTHROPIC_BASE_URL` at.
   It routes each request by model name. Requests Claude Code sends to its `haiku` slot go to
   **DeepSeek** (which serves an Anthropic-compatible API), and Opus requests go to Anthropic.
   It writes a per-call usage and cost ledger, which is how every number below was measured.

## Results

A toy autoresearch task runs on CPU in [`bench/`](bench/). The task is byte-level language
modelling on Python source with a 60 s training budget. An agent gets a baseline plus 4
experiments, each with a fixed protocol: e2e check, submit to a simulated cluster queue, wait,
read metrics, log, git keep/discard. Each run is a real headless Claude Code session. Opus 5.5
does the research. The cheap slot was **Haiku 4.5**, because this environment has no DeepSeek
key, and every one of those calls is also repriced at DeepSeek V4.1 Flash rates from its exact
token counts. The labour load was varied:

- **light**: a tidy queue with a `result` command. Jobs finish inside one shell call, so a
  single `until …; do sleep 15; done` loop covers the whole wait.
- **heavy**: a busy shared cluster. Each experiment needs 2 seeds, there is no `result` command
  (metrics must be read from the logs), logs are spammed with heartbeats, and 2 node failures
  force resubmits.
- **long jobs**: the heavy cluster, plus every shell command is killed after 30 s. This is how
  real autoresearch looks: training jobs last minutes to hours, but a Claude Code tool call
  can wait at most 10 minutes, so the agent has to come back and poll repeatedly.

![cost by arm](results/cost_by_arm.png)

Mean $ per run (2 runs per cell), measured through the router:

| condition | | Opus alone | Opus + operator | Δ |
|---|---|---|---|---|
| light | Opus $ (labour share) | 0.37 (61%) | 0.31 (0%) | **−16%** Opus |
| | total $, cheap = DeepSeek V4.1 Flash | 0.37 | 0.34 | −8% |
| | total $, cheap = Haiku 4.5 | 0.37 | 0.45 | +22% |
| heavy | Opus $ (labour share) | 0.52 (63%) | 0.42 (3%) | **−19%** Opus |
| | total $, cheap = DeepSeek | 0.52 | 0.47 | −10% |
| | total $, cheap = Haiku | 0.52 | 0.67 | +29% |
| **long jobs** | Opus $ (labour share) | 0.96 (**80%**) | 0.39 (4%) | **−59%** Opus |
| | total $, cheap = DeepSeek | 0.96 | **0.46** | **−52%** |
| | total $, cheap = Haiku | 0.96 | 0.81 | −16% |

Best val_bpb (lower is better; baseline ≈ 3.30) over the same 6 runs per arm:
**Opus alone 2.359 ± 0.048**, **Opus + operator 2.395 ± 0.036**. That difference is not
statistically significant (Welch t = 1.5). Both arms found the same main ideas (AdamW, a
higher LR), and Opus thinks just as well when it reads a short report instead of logs.
Possibly a small cost remains, because the report hides the loss curve. The cheap model on
its own reached only **2.79** (2.57 / 3.02), so it cannot stand in for Opus on the thinking.

Full tables: [`results/REPORT.md`](results/REPORT.md). Raw ledgers and transcripts:
[`results/runs/`](results/runs/).

### What the experiments say

1. **Your ~50% figure is real.** Opus alone spent 61% / 63% / 80% of its dollars on labour
   in the light / heavy / long-jobs conditions. With the operator, Opus's labour share drops
   to 0-4% in every condition.
2. **The savings depend on how many Opus turns the labour took, not on token type.** Every
   Opus turn costs about **$0.02** at a 15-30k-token context, whatever it does: re-reading
   the context, writing the new tool output into the cache, a few hundred output tokens.
   Delegating still costs Opus a hand-off turn per experiment, on top of the turn that edits
   the code and reads the previous report. Hand-offs plus Claude Code's own calls were about
   30% of Opus's remaining spend. So when Opus could already do
   check, submit and wait in one shell command (light, heavy), delegation saved only 16-19% of
   Opus spend. When jobs outlast a tool call and Opus has to keep polling (long jobs: 56 Opus
   calls a run instead of 22), it saved **59%**. Opus's bill then stops depending on how long
   the jobs run.
3. **The cheap model has to be really cheap.** At Haiku prices ($1/$5, $2/M for 1-hour cache
   writes) the operator's own polling cost more than it saved, except in the long-jobs
   condition. At DeepSeek V4.1 Flash prices ($0.30/$1.20, $0.006/M cache hits) labour is
   almost free, and the end-to-end saving is close to the Opus saving: **−52%** with long jobs.
4. **Keep Opus as the main loop. Don't make it a stateless subagent.** The inverted design
   (cheap driver, Opus `researcher` subagent) had the best val_bpb (2.327, n=2) but the
   *highest* Opus bill ($0.69-0.75 a run, versus $0.37-0.52 for Opus alone). Every researcher
   call starts cold: it re-reads the program, code and notes, pays the cache write again and
   re-thinks from scratch. It is also fragile. Prompt rules alone failed: the Haiku driver
   skipped the researcher and wrote all the code itself, and that run scored 3.18. A PreToolUse
   hook ([`only_researcher_edits.py`](claude/hooks/only_researcher_edits.py)) had to enforce
   the rule. After that, the driver still dictated the ideas until the prompts forbade it.
5. **Cost is mostly the input side, not output.** Opus's bill splits as ~40% output,
   ~40% cache writes and ~20% cache reads; uncached input is ~2%. See the next section.

### What to do in your autoresearch

- If your jobs run longer than one tool call (minutes to hours), use **Opus as the main loop
  plus an `operator` subagent on DeepSeek**. That is the long-jobs setting, where this halves
  the bill without hurting research quality.
- Have Opus hand over a whole experiment, or several, in one operator call, so delegation
  overhead stays at one hand-off turn per experiment.
- The cheapest labour is labour no model does. A blocking `wait` command, or one polling loop
  inside a single shell call, removes most polling turns for any model. That is why delegation
  saved little in the light condition.

## Why is input cheaper than output?

On Opus 5.5, input costs $4/M and output $20/M, 5x more. DeepSeek charges about 4x more for output.
The reason is how the GPU works in each phase:

- **Input (prefill) runs in parallel.** All N prompt tokens go through the model in one
  forward pass as large matrix-matrix multiplies, so the GPU is compute-bound and nearly
  fully used. Each weight loaded from memory serves thousands of tokens.
- **Output (decode) runs one token at a time.** Every new token needs a full forward pass
  that reads all the weights, plus that sequence's whole KV cache, from GPU memory to
  produce a single token. That is memory-bandwidth-bound: the arithmetic units mostly sit
  idle. Providers batch many users' decode steps together to share the weight reads, but the
  batch size is limited by KV-cache memory, because every in-flight sequence keeps its whole
  context resident.
- **Price follows GPU-seconds per token.** An output token holds a batch slot and its KV
  memory for a full sequential step. An input token is a small slice of one parallel pass.
- **Cached input is cheaper still** ($0.20/M on Opus 5.5, 0.05x): its KV is already computed,
  so the provider only loads it. Writing the cache costs extra: 1.25x input for a 5-minute
  TTL, 2x for 1 hour (Claude Code uses the 1-hour TTL).

### What that means for agent loops (a correction to the premise)

The hope was that moving the labour would turn expensive Opus output tokens into cheap
input tokens. That is not quite what happens. Every agent turn re-sends the whole
conversation as input. A turn that only polls the queue writes ~150 output tokens, but it
also re-reads 15-30k tokens of context and writes the new log output into the cache at $8/M.
In the runs above that came to ~$0.02 per turn, of which output was under half. Orchestration
is expensive because of **how many turns it takes × how big the context is**, and because the
logs it reads pile into the context that every later Opus turn re-reads.

So the saving does not come from converting token types. It comes from **Opus not taking
those turns at all**. It sees an 8-line report instead of logs, so its context also stays
small. The labour itself is then billed at the cheap model's rates. For DeepSeek V4.1 Flash
that is 13-33x below Opus on fresh input and output and 33-67x below on cache hits
(peak / off-peak).

## Use it on your own project

```bash
# 1. the division of labour
mkdir -p .claude/agents && cp claude/agents/operator.md .claude/agents/   # adapt its steps to your pipeline
cat claude/CLAUDE.delegate.md >> CLAUDE.md

# 2. route the cheap slot to DeepSeek (Opus keeps going to Anthropic)
export DEEPSEEK_API_KEY=...
python -m marx.router --config marx/routes.deepseek.json --port 8787 --ledger ledger.jsonl &
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude --model opus

# 3. see where the money went
python -m marx.report ledger.jsonl
python -m marx.report ledger.jsonl --as-if 'claude-haiku-*=deepseek-flash'   # what-if repricing
```

`routes.deepseek.json` sends every `claude-haiku-*` request to
`https://api.deepseek.com/anthropic` as `deepseek-flash`, using your DeepSeek key. Everything
else passes through to Anthropic with your normal credentials. The `operator` subagent declares
`model: haiku`, so its traffic goes to DeepSeek. Claude Code's own background calls on the haiku
slot (titles, summaries) go there too. To use a different cheap provider, change `upstream`,
`model` and `api_key_env`. A route can also set `drop_fields` (request keys to strip) and
`drop_betas` for upstreams that reject Anthropic-only parameters. `routes.anthropic.json`
routes everything to Anthropic, which gives you the ledger alone.

Caveat: the DeepSeek route was not exercised end to end here, because this environment had no
DeepSeek key. The router itself carried every request in the experiments. DeepSeek's tokenizer
will also count somewhat differently from the repriced Haiku token counts.

## Reproduce

```bash
pip install torch numpy matplotlib           # CPU torch is enough
python bench/prepare.py                      # builds the corpus
python experiments/run_batch.py --jobs 4 \
  light:solo:1 light:delegate:1 heavy:solo:1 heavy:delegate:1 long:solo:1 long:delegate:1
python experiments/analyze.py && python experiments/plot.py
```

Arms: `solo`, `delegate`, `inverted`, `cheap`. Conditions: `light`, `heavy`, `long`. Each run
drives the logged-in `claude` CLI headless, costs real money ($0.2-1.1 per run here), and
takes 12-30 minutes.

## Layout

| path | what |
|---|---|
| `marx/router.py` | routing proxy and usage ledger |
| `marx/prices.py`, `marx/report.py` | price table; ledger summary with `--as-if` repricing |
| `claude/agents/operator.md` | cheap-model subagent: check, submit, wait, resubmit, record, keep/discard |
| `claude/agents/researcher.md` | Opus subagent for the inverted form: choose the next idea, edit `train.py`, keep a lab notebook |
| `claude/hooks/only_researcher_edits.py` | PreToolUse hook enforcing the inverted form |
| `claude/CLAUDE.delegate.md`, `claude/CLAUDE.inverted.md` | system-prompt snippets for the two topologies |
| `bench/` | the toy autoresearch task and simulated cluster (`cluster.json` picks light/heavy) |
| `experiments/` | arm runner, batch runner, cost attribution, plot |
| `results/` | `REPORT.md`, chart, and per run: ledger, transcript, `results.tsv`, final `train.py` |
