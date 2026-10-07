"""Invisible model switching: one conversation, the router decides who answers each turn.

The agent runs with a single set of instructions and never hears about modes or models. Before
each main-loop request the dispatcher picks the model for the *next* turn:

  think   (expensive, e.g. Opus): judgment - interpreting results, choosing the next experiment,
          designing / writing / fixing code, the final summary
  labour  (cheap, e.g. Haiku 5.5): routine execution - running checks and scripts, submitting jobs,
          waiting and polling, resubmitting after infrastructure failures, fetching logs

The call is made by a small classifier prompt on the cheap model (fractions of a cent per turn),
plus two fixed rules: the first turn and any turn that follows a plain-text user message go to
`think`. The dispatcher remembers which model produced which assistant turn, so it can

  * re-anchor each model's prompt cache where that model's previous request ended, and
  * (optional) show the expensive model each stretch of routine turns compressed to its last
    step plus a one-line digest of the commands that ran ("elide"). The digest is built the same
    way on every request, so the expensive model's view of history never changes after it has
    seen it (prompt cache and preserved-thinking checks stay valid).
"""
import hashlib
import json
import re
import threading

CLASSIFIER_SYSTEM = """You route the turns of an autonomous coding/research agent between two models.

EXPENSIVE takes judgment calls: reading new results and deciding what they mean, choosing what to try
next, designing, writing or fixing code, debugging a crash or a failed check, deciding keep/discard,
writing the final summary.
CHEAP takes routine execution whose next action is already clear: running a check or script,
submitting jobs, waiting for jobs and polling their status, resubmitting after an infrastructure
failure, fetching logs, running a command the agent has just decided on.

You see the agent's most recent steps. Decide who should take the NEXT step.
If the latest output contains a new final metric, a failure, or anything that needs interpreting,
answer EXPENSIVE. If unsure, answer EXPENSIVE.
Answer with exactly one word: EXPENSIVE or CHEAP."""

TOOL_INPUT_KEYS = ("command", "file_path", "pattern", "description", "prompt")


def _blocks(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])


def _text_of(block, limit):
    if block.get("type") == "text":
        s = block.get("text", "")
    elif block.get("type") == "tool_use":
        inp = block.get("input") or {}
        arg = next((str(inp[k]) for k in TOOL_INPUT_KEYS if k in inp), json.dumps(inp)[:200])
        s = f"[tool call {block.get('name')}] {arg}"
    elif block.get("type") == "tool_result":
        c = block.get("content")
        c = c if isinstance(c, str) else " ".join(b.get("text", "") for b in c or [] if isinstance(b, dict))
        s = f"[tool result] {c}"
    else:
        return ""
    s = s.strip()
    return s if len(s) <= limit else s[: limit // 3] + " ... " + s[-(2 * limit) // 3:]


def digest_recent(messages, n_messages=6, limit=900):
    """Compact text view of the last few messages, for the classifier."""
    lines = []
    for msg in messages[-n_messages:]:
        if msg.get("role") not in ("user", "assistant"):
            continue
        parts = [t for t in (_text_of(b, limit) for b in _blocks(msg.get("content"))) if t]
        if parts:
            lines.append(f"{msg['role'].upper()}: " + "\n  ".join(parts))
    return "\n".join(lines)


class Conversation:
    def __init__(self):
        self.served_by = {}     # index of an assistant message -> mode that produced it
        self.last_len = {}      # mode -> number of messages in that mode's latest request
        self.lock = threading.Lock()


class Dispatcher:
    def __init__(self, spec, classify_fn):
        self.elide = spec.get("elide", False)
        self.keep_last = spec.get("keep_last", 1)
        self.lookback = spec.get("lookback", 15)
        self.classify_fn = classify_fn          # (system, user_text) -> "think" | "labour"
        self.conversations = {}
        self.lock = threading.Lock()

    def conversation(self, payload):
        meta = (payload.get("metadata") or {}).get("user_id", "")
        first = json.dumps((payload.get("messages") or [{}])[0], sort_keys=True)[:4000]
        key = hashlib.sha256((meta + first).encode()).hexdigest()
        with self.lock:
            return self.conversations.setdefault(key, Conversation())

    def choose(self, payload):
        """Mode for the next turn of this main-loop request."""
        msgs = payload.get("messages") or []
        if not any(m.get("role") == "assistant" for m in msgs):
            return "think", "rule:first-turn"
        last = msgs[-1]
        if last.get("role") == "user" and not any(b.get("type") == "tool_result" for b in _blocks(last.get("content"))):
            return "think", "rule:user-message"
        verdict = self.classify_fn(CLASSIFIER_SYSTEM, digest_recent(msgs))
        return verdict, "classifier"

    def prepare(self, conv, payload, mode):
        """Rewrite the messages this mode's model will see; record who answers this request."""
        msgs = payload["messages"]
        n = len(msgs)
        view = msgs
        if mode == "think" and self.elide:
            view = self._elide(conv, msgs)
        else:
            self._anchor(conv, view, mode)
        conv.served_by[n] = mode          # the reply will be message n
        conv.last_len[mode] = n
        payload["messages"] = view

    def _elide(self, conv, msgs):
        """Collapse each run of labour-served assistant turns (and their tool results) to its last
        `keep_last` turns, with a digest of what ran appended to the message before the run."""
        out, i = [], 0
        while i < len(msgs):
            if msgs[i].get("role") == "assistant" and conv.served_by.get(i) == "labour":
                j = i
                while j < len(msgs) and (msgs[j].get("role") != "assistant" or conv.served_by.get(j) == "labour"):
                    j += 1
                run = msgs[i:j]                       # labour turns + their results, up to the next think turn
                starts = [k for k, m in enumerate(run) if m.get("role") == "assistant"]
                cut = starts[-self.keep_last] if len(starts) > self.keep_last else 0
                if cut and out and out[-1].get("role") == "user":
                    note = digest_elided(run[:cut], len(starts) - self.keep_last)
                    out[-1] = dict(out[-1], content=_blocks(out[-1]["content"]) + [{"type": "text", "text": note}])
                    out.extend(run[cut:])
                else:
                    out.extend(run)
                i = j
            else:
                out.append(msgs[i])
                i += 1
        return out

    def _anchor(self, conv, msgs, mode):
        """Move Claude Code's second message breakpoint to where this mode's previous request ended,
        if that is beyond the API's ~20-block cache lookback."""
        prev = conv.last_len.get(mode)
        if not prev or prev > len(msgs):
            return
        target = prev - 1
        blocks_after = sum(len(_blocks(m.get("content"))) for m in msgs[target + 1:])
        if blocks_after <= self.lookback or not isinstance(msgs[target].get("content"), list) \
                or not msgs[target]["content"]:
            return
        marked = [b for m in msgs if isinstance(m.get("content"), list)
                  for b in m["content"] if isinstance(b, dict) and "cache_control" in b]
        if not marked:
            return
        cc = dict(marked[-1]["cache_control"])
        if len(marked) >= 2:
            marked[0].pop("cache_control")
        msgs[target]["content"][-1]["cache_control"] = cc


def _result_tail(block, lines=3, limit=240):
    c = block.get("content")
    c = c if isinstance(c, str) else "\n".join(b.get("text", "") for b in c or [] if isinstance(b, dict))
    tail = " | ".join([ln.strip() for ln in c.strip().splitlines() if ln.strip()][-lines:])
    return tail if len(tail) <= limit else "..." + tail[-limit:]


def digest_elided(run, n_steps, max_cmds=12):
    """One text block standing in for elided routine turns: each distinct command once, with how
    often it ran and the tail of its latest output (so final states and metrics survive)."""
    results = {}
    for m in run:
        for b in _blocks(m.get("content")):
            if b.get("type") == "tool_result":
                results[b.get("tool_use_id")] = b
    order, count, last_out = [], {}, {}
    for m in run:
        for b in _blocks(m.get("content")):
            if b.get("type") != "tool_use":
                continue
            inp = b.get("input") or {}
            cmd = str(next((inp[k] for k in TOOL_INPUT_KEYS if k in inp), b.get("name"))).replace("\n", " ")[:90]
            if cmd not in count:
                order.append(cmd)
            count[cmd] = count.get(cmd, 0) + 1
            if b.get("id") in results:
                last_out[cmd] = _result_tail(results[b["id"]])
    items = [f"`{c}`{f' (x{count[c]})' if count[c] > 1 else ''} -> {last_out.get(c, '')}" for c in order[-max_cmds:]]
    more = f" ({len(order) - max_cmds} more commands not listed)" if len(order) > max_cmds else ""
    return (f"[{n_steps} earlier routine steps condensed in this view{more}; latest output of each:\n- "
            + "\n- ".join(items) + "]")


def parse_verdict(text):
    t = (text or "").upper()
    if "CHEAP" in t and "EXPENSIVE" not in t:
        return "labour"
    return "think"
