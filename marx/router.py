"""Model-routing proxy for Claude Code (or anything speaking the Anthropic Messages API).

Claude Code sends every request -- main loop and subagents -- to ANTHROPIC_BASE_URL with
the model name in the JSON body. This proxy sits at that URL and forwards each request to
an upstream chosen by model name, so e.g. everything Claude Code sends as "haiku" can be
served by DeepSeek (which speaks the Anthropic API) while Opus traffic goes to Anthropic.
Every call's token usage and cost is appended to a JSONL ledger.

A route can also *switch models within one conversation*: with a "switch" block, each
main-loop request is served by the model of the current mode, and the mode is whatever the
agent last set with the shell command `marx-mode <mode>` (read back from the forwarded
history). So a cheap model can take the labour turns and Opus the thinking turns of the SAME
chat, each seeing the full history. See marx/routes.switch.*.json and claude/CLAUDE.switch.md.

    python -m marx.router --config marx/routes.deepseek.json --port 8787 --ledger ledger.jsonl
    ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude ...

Stdlib only. Streams SSE through unchanged.
"""
import argparse
import fnmatch
import http.client
import json
import os
import re
import ssl
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from marx.prices import usage_cost

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
              "trailers", "transfer-encoding", "upgrade", "host", "content-length",
              "accept-encoding"}


MODE_RE = re.compile(r"marx-mode\s+(\w+)")


class Route:
    def __init__(self, spec, name=None):
        self.name = name or spec.get("name") or spec.get("match")
        self.match = spec.get("match", "*")
        self.upstream = urllib.parse.urlsplit(spec["upstream"]) if "upstream" in spec else None
        self.api_key_env = spec.get("api_key_env")   # None -> pass client auth through
        self.model = spec.get("model")               # rewrite the model name
        self.drop_fields = spec.get("drop_fields", [])
        self.drop_betas = spec.get("drop_betas", False)
        self.strip_thinking = spec.get("strip_thinking", False)  # for upstreams that can't take other models' thinking
        self.system_append = spec.get("system_append")
        self.max_tokens = spec.get("max_tokens")     # cap: the client may ask for more than this model allows
        self.fold_system_messages = spec.get("fold_system_messages", False)  # for models without mid-chat system
        # Reminder appended to the newest user message. Only for models without Opus-style
        # preserved-thinking checks: the next request sends that message without the note,
        # which those models would treat as an edit of history.
        self.turn_note = spec.get("turn_note")
        # Show this model the other mode's streaks only as the hand-back message (deterministically,
        # so its cache and preserved-thinking checks are unaffected): what it skips is labour it
        # never saw, after its own last turn.
        self.elide_other_modes = spec.get("elide_other_modes", False)
        # Same-conversation model switching: the conversation itself says who acts next, via the
        # last `marx-mode <mode>` shell command in the history. Each mode is a route of its own.
        sw = spec.get("switch")
        self.modes = {m: Route(sub, f"{self.name}:{m}") for m, sub in sw["modes"].items()} if sw else None
        self.mode = None
        for m, sub in (self.modes or {}).items():
            sub.mode = m
        self.default_mode = sw.get("default") if sw else None

    def matches(self, model):
        return fnmatch.fnmatch(model or "", self.match)

    def resolve(self, payload):
        """The concrete route for this request (a mode sub-route when switching)."""
        if not self.modes or not isinstance(payload, dict) or not payload.get("tools"):
            return self.modes[self.default_mode] if self.modes else self
        return self.modes.get(last_mode(payload.get("messages") or []) or self.default_mode,
                              self.modes[self.default_mode])

    def rewrite(self, payload):
        if self.model:
            payload["model"] = self.model
        for f in self.drop_fields:
            payload.pop(f, None)
        if self.max_tokens and (payload.get("max_tokens") or 0) > self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if self.strip_thinking:
            for msg in payload.get("messages") or []:
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                    kept = [b for b in msg["content"] if b.get("type") not in ("thinking", "redacted_thinking")]
                    msg["content"] = kept or [{"type": "text", "text": "."}]
        if self.fold_system_messages and payload.get("messages"):
            payload["messages"] = fold_system_messages(payload["messages"])
        if self.mode and self.elide_other_modes and payload.get("messages"):
            payload["messages"] = elide_other_modes(payload["messages"], self.mode)
        if self.mode and payload.get("messages"):
            anchor_cache(payload["messages"], self.mode)
        if self.turn_note and payload.get("messages") and payload["messages"][-1].get("role") == "user":
            last = payload["messages"][-1]
            last["content"] = _blocks(last["content"]) + [
                {"type": "text", "text": f"<system-reminder>{self.turn_note}</system-reminder>"}]
        if self.system_append:
            sysp = payload.get("system")
            block = {"type": "text", "text": self.system_append}
            if isinstance(sysp, list):
                sysp.append(block)
            else:
                payload["system"] = [{"type": "text", "text": sysp}, block] if sysp else [block]


def _blocks(content):
    return [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])


def fold_system_messages(messages):
    """Turn mid-conversation `role: system` messages into user text and merge same-role
    neighbours, for upstreams that only accept alternating user/assistant turns."""
    out = []
    for msg in messages:
        role, blocks = msg.get("role"), _blocks(msg.get("content"))
        if role == "system":
            text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            if not text:          # e.g. an effort-only operator message
                continue
            role, blocks = "user", [{"type": "text", "text": f"<system-reminder>\n{text}\n</system-reminder>"}]
        if out and out[-1]["role"] == role:
            out[-1]["content"] = _blocks(out[-1]["content"]) + blocks
        else:
            out.append(dict(msg, role=role, content=blocks))
    return out


def _mode_switches(msg):
    for block in msg.get("content") if isinstance(msg.get("content"), list) else []:
        if block.get("type") == "tool_use":
            m = MODE_RE.search(str((block.get("input") or {}).get("command", "")))
            if m:
                yield m.group(1)


def elide_other_modes(messages, mode):
    """Drop the turns another mode took between a hand-off (`marx-mode <other>`) and the hand-back
    (`marx-mode <mode>`), keeping the hand-off's tool result and the hand-back message itself."""
    out, i = [], 0
    while i < len(messages):
        msg = messages[i]
        out.append(msg)
        leaves = msg.get("role") == "assistant" and any(m != mode for m in _mode_switches(msg))
        if leaves and i + 1 < len(messages):
            back = next((j for j in range(i + 2, len(messages))
                         if messages[j].get("role") == "assistant" and mode in _mode_switches(messages[j])), None)
            if back is None:          # still inside the other mode's streak
                i += 1
                continue
            out.append(messages[i + 1])               # tool result of the hand-off
            n = back - (i + 2)
            if n > 0:
                out[-1] = dict(out[-1], content=_blocks(out[-1]["content"]) + [{"type": "text", "text":
                          f"[marx: {n} messages of the other model's turns are elided here; its hand-back report follows]"}])
            i = back
            continue
        i += 1
    return out


def anchor_cache(messages, mode, lookback=15):
    """Re-anchor a prompt-cache breakpoint where this mode's model last left off.

    The API finds an earlier cached prefix only within ~20 content blocks of a breakpoint, and
    Claude Code puts its message breakpoints on the last two messages. After a long streak of the
    other model's turns, the current model's own cached prefix is out of reach and the whole
    conversation would be written to its cache again. Its previous request ended right before
    the message in which it handed over, so move Claude Code's second message breakpoint there.
    """
    handoff = next((i for i in range(len(messages) - 1, -1, -1)
                    if messages[i].get("role") == "assistant"
                    and any(m != mode for m in _mode_switches(messages[i]))), None)
    if not handoff:
        return
    target = handoff - 1
    blocks_after = sum(len(m["content"]) if isinstance(m.get("content"), list) else 1
                       for m in messages[target + 1:])
    if blocks_after <= lookback or not isinstance(messages[target].get("content"), list) \
            or not messages[target]["content"]:
        return
    marked = [(i, b) for i, m in enumerate(messages) if isinstance(m.get("content"), list)
              for b in m["content"] if isinstance(b, dict) and "cache_control" in b]
    if not marked:
        return
    cc = dict(marked[-1][1]["cache_control"])
    if len(marked) >= 2:                       # free a breakpoint: the max is 4 per request
        marked[0][1].pop("cache_control")
    messages[target]["content"][-1]["cache_control"] = cc


def last_mode(messages):
    """Mode named by the most recent `marx-mode <mode>` Bash call in the conversation."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant" or not isinstance(msg.get("content"), list):
            continue
        for block in reversed(msg["content"]):
            if block.get("type") == "tool_use":
                m = MODE_RE.search(str((block.get("input") or {}).get("command", "")))
                if m:
                    return m.group(1)
    return None


class Router:
    def __init__(self, config, ledger_path):
        self.routes = [Route(r) for r in config["routes"]]
        self.ledger_path = ledger_path
        self.lock = threading.Lock()
        self.ctx = ssl.create_default_context(
            cafile=os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE") or None)

    def pick(self, model):
        for r in self.routes:
            if r.matches(model):
                return r
        raise LookupError(f"no route for model {model!r}")

    def connect(self, url):
        host, port = url.hostname, url.port or (443 if url.scheme == "https" else 80)
        if url.scheme == "http":
            return http.client.HTTPConnection(host, port, timeout=600)
        proxy = urllib.request.getproxies().get("https")
        if proxy and not urllib.request.proxy_bypass(host):
            p = urllib.parse.urlsplit(proxy)
            conn = http.client.HTTPSConnection(p.hostname, p.port, timeout=600, context=self.ctx)
            conn.set_tunnel(host, port)
            return conn
        return http.client.HTTPSConnection(host, port, timeout=600, context=self.ctx)

    def record(self, entry):
        if not self.ledger_path:
            return
        with self.lock, open(self.ledger_path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")


def make_handler(router):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # keep stderr quiet; the ledger is the log
            pass

        def _proxy(self):
            length = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(length) if length else b""
            payload = None
            if body and "json" in (self.headers.get("content-type") or ""):
                try:
                    payload = json.loads(body)
                except ValueError:
                    payload = None
            requested = payload.get("model") if isinstance(payload, dict) else None
            try:
                route = router.pick(requested or "").resolve(payload)
            except LookupError as e:
                return self._error(404, str(e))

            if isinstance(payload, dict):
                route.rewrite(payload)
                body = json.dumps(payload).encode()
                if os.environ.get("MARX_CAPTURE_DIR"):  # debugging: keep every forwarded request body
                    with router.lock:
                        router.n_captured = getattr(router, "n_captured", 0) + 1
                        n = router.n_captured
                    with open(os.path.join(os.environ["MARX_CAPTURE_DIR"], f"{n:04d}-{route.name}.json"), "wb") as fh:
                        fh.write(body)

            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            headers["Accept-Encoding"] = "identity"   # so the ledger can read usage from the stream
            headers["Content-Length"] = str(len(body))
            if route.api_key_env:
                for k in list(headers):
                    if k.lower() in ("authorization", "x-api-key"):
                        del headers[k]
                headers["x-api-key"] = os.environ.get(route.api_key_env, "")
            if route.drop_betas:
                headers = {k: v for k, v in headers.items() if k.lower() != "anthropic-beta"}

            path = route.upstream.path.rstrip("/") + self.path
            t0 = time.time()
            try:
                conn = router.connect(route.upstream)
                conn.request(self.command, path, body=body, headers=headers)
                resp = conn.getresponse()
            except OSError as e:
                return self._error(502, f"upstream {route.upstream.netloc} unreachable: {e}")

            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            sse = "text/event-stream" in (resp.getheader("content-type") or "")
            usage, resp_model, msg_id, buf, raw = {}, None, None, b"", b""
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                if sse:
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for line in lines:
                        if not line.startswith(b"data:"):
                            continue
                        try:
                            ev = json.loads(line[5:])
                        except ValueError:
                            continue
                        if ev.get("type") == "message_start":
                            resp_model = ev["message"].get("model")
                            msg_id = ev["message"].get("id")
                            usage.update(ev["message"].get("usage") or {})
                        elif ev.get("type") == "message_delta":
                            usage.update({k: v for k, v in (ev.get("usage") or {}).items() if v is not None})
                elif len(raw) < 4_000_000:
                    raw += chunk
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            conn.close()

            if not sse and raw:
                try:
                    j = json.loads(raw)
                    usage, resp_model, msg_id = j.get("usage") or {}, j.get("model"), j.get("id")
                except ValueError:
                    pass
            if usage and self.path.split("?")[0].endswith("/messages"):
                served = route.model or requested
                try:
                    cost = usage_cost(served, usage)
                except KeyError:
                    cost = None
                router.record({"ts": t0, "latency_s": round(time.time() - t0, 3), "route": route.name,
                               "requested_model": requested, "served_model": served,
                               "upstream_reported_model": resp_model, "message_id": msg_id,
                               "status": resp.status,
                               "usage": usage, "cost_usd": cost})

        def _error(self, code, msg):
            data = json.dumps({"type": "error", "error": {"type": "proxy_error", "message": msg}}).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_POST = do_GET = do_PUT = do_DELETE = do_PATCH = _proxy

    return Handler


class QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            super().handle_error(request, client_address)


def build_server(config_path, port, ledger):
    with open(config_path) as fh:
        config = json.load(fh)
    router = Router(config, ledger)
    srv = QuietServer(("127.0.0.1", port), make_handler(router))
    print(f"marx router on http://127.0.0.1:{srv.server_port}  routes: "
          + ", ".join(f"{r.match}->" + ("switch(" + ", ".join(f"{m}:{sub.upstream.netloc}/{sub.model or 'same'}"
                                                             for m, sub in r.modes.items()) + ")" if r.modes
                                        else f"{r.upstream.netloc}{'/' + r.model if r.model else ''}")
                      for r in router.routes),
          file=sys.stderr, flush=True)
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--ledger", default="ledger.jsonl")
    a = ap.parse_args()
    build_server(a.config, a.port, a.ledger).serve_forever()


if __name__ == "__main__":
    main()
