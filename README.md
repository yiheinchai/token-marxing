# token-marxing

> *From each model according to its price, to each task according to its difficulty.*

When Claude Code runs an autoresearch loop, Opus spends most of its tokens on **labour**:
running `.py` files, e2e checks, queueing jobs, waiting for jobs, reading logs, `git`.
token-marxing gives that labour to a cheap model (DeepSeek V4.1 Flash, Haiku, ...) and
keeps Opus for the **intellectual work**: deciding what to try and writing the PyTorch code.

It has two parts:

1. **A division of labour inside Claude Code**: an `operator` subagent on the cheap slot
   does the pipeline work, or (the inverted form) the cheap model drives and calls a
   `researcher` subagent on Opus whenever code has to change. See [`claude/`](claude/).
2. **`marx/router.py`**: a stdlib-only proxy you point `ANTHROPIC_BASE_URL` at. It routes
   each request by model name, so the requests Claude Code sends to its `haiku` slot go to
   **DeepSeek** (which serves an Anthropic-compatible API) while Opus requests go to Anthropic.
   It also writes a per-call usage and cost ledger, which is how the numbers below were measured.

RESULTS_PLACEHOLDER

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

### What that means for agent loops (and a correction to the premise)

In an agent loop the expensive part is usually **not** output tokens. Every turn re-sends the
whole conversation as input. A turn that only says "poll the queue again" writes about 100
output tokens ($0.002), but it re-reads 30-60k tokens of context ($0.006-0.012 at cache-read
prices) and writes the new log output into the cache at $8/M. Orchestration is expensive
because of **how many turns it takes × how big the context is**, and because raw logs pile
into the context that every later Opus turn re-reads.

So moving the labour to a cheap model does not just turn Opus output into Opus input.
The real effect is that **Opus stops taking those turns at all**. It sees an 8-line report
instead of logs, so its context also stays small. The labour itself is then billed at the
cheap model's rates, which for DeepSeek V4.1 Flash are 13-30x below Opus on fresh input and
output and about 33x below on cache reads.

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
```

`routes.deepseek.json` sends every `claude-haiku-*` request to
`https://api.deepseek.com/anthropic` as `deepseek-flash`, using your DeepSeek key, and passes
everything else through to Anthropic with your normal credentials. Because the `operator`
subagent declares `model: haiku`, its traffic goes to DeepSeek. Claude Code's own
background calls on the haiku slot (titles, summaries) go there too. To use a different
cheap provider, change `upstream`, `model` and `api_key_env`. Each route also accepts
`drop_fields` (request keys to strip) and `drop_betas` for upstreams that reject
Anthropic-only parameters. With no `--config` changes, `routes.anthropic.json` routes
everything to Anthropic and you get the ledger alone.

## Reproduce the experiment

```bash
pip install torch numpy matplotlib          # CPU torch is enough
python bench/prepare.py                     # builds the corpus
python experiments/run_arm.py --arm solo     --seed 1 --port 8801
python experiments/run_arm.py --arm delegate --seed 1 --port 8802
python experiments/run_arm.py --arm inverted --seed 1 --port 8803
python experiments/run_arm.py --arm cheap    --seed 1 --port 8804
python experiments/analyze.py && python experiments/plot.py
```

Each run uses the logged-in `claude` CLI and costs real money.

## Layout

| path | what |
|---|---|
| `marx/router.py` | routing proxy and usage ledger |
| `marx/prices.py`, `marx/report.py` | price table; ledger summary with `--as-if` repricing |
| `claude/agents/operator.md` | cheap-model subagent: check, submit, wait, resubmit, record, keep/discard |
| `claude/agents/researcher.md` | Opus subagent: choose the next idea, edit `train.py`, keep a lab notebook |
| `claude/CLAUDE.delegate.md`, `claude/CLAUDE.inverted.md` | system-prompt snippets for the two topologies |
| `bench/` | the toy autoresearch task |
| `experiments/` | arm runner, cost attribution, plot |
| `results/` | ledgers, transcripts, the agents' `results.tsv` and final `train.py` for every run |
