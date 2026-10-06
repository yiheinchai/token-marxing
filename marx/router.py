"""Model-routing proxy for Claude Code (or anything speaking the Anthropic Messages API).

Claude Code sends every request -- main loop and subagents -- to ANTHROPIC_BASE_URL with
the model name in the JSON body. This proxy sits at that URL and forwards each request to
an upstream chosen by model name, so e.g. everything Claude Code sends as "haiku" can be
served by DeepSeek (which speaks the Anthropic API) while Opus traffic goes to Anthropic.
Every call's token usage and cost is appended to a JSONL ledger.

    python -m marx.router --config marx/routes.deepseek.json --port 8787 --ledger ledger.jsonl
    ANTHROPIC_BASE_URL=http://127.0.0.1:8787 claude ...

Stdlib only. Streams SSE through unchanged.
"""
import argparse
import fnmatch
import http.client
import json
import os
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


class Route:
    def __init__(self, spec):
        self.name = spec.get("name") or spec["match"]
        self.match = spec["match"]
        self.upstream = urllib.parse.urlsplit(spec["upstream"])
        self.api_key_env = spec.get("api_key_env")   # None -> pass client auth through
        self.model = spec.get("model")               # rewrite the model name
        self.drop_fields = spec.get("drop_fields", [])
        self.drop_betas = spec.get("drop_betas", False)

    def matches(self, model):
        return fnmatch.fnmatch(model or "", self.match)


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
                route = router.pick(requested or "")
            except LookupError as e:
                return self._error(404, str(e))

            if isinstance(payload, dict):
                if route.model:
                    payload["model"] = route.model
                for f in route.drop_fields:
                    payload.pop(f, None)
                body = json.dumps(payload).encode()

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
          + ", ".join(f"{r.match}->{r.upstream.netloc}{'/' + r.model if r.model else ''}" for r in router.routes),
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
