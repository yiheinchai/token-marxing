#!/usr/bin/env python3
"""PreToolUse hook: only the `researcher` subagent may change train.py.

Used by the inverted topology, where a cheap model drives the loop. Cheap drivers tend to
"just do it themselves" instead of delegating the thinking (observed in our first inverted
run), so the rule is enforced here rather than left to the prompt. Exit code 2 blocks the
tool call and shows stderr to the model.
"""
import json
import re
import sys

PROTECTED = "train.py"
WRITES = re.compile(r"(sed\s+-i|>\s*\S*train\.py|tee\s+\S*train\.py|cp\s+\S+\s+\S*train\.py|"
                    r"mv\s+\S+\s+\S*train\.py|open\([^)]*train\.py[^)]*['\"]w|patch\b|perl\s+-[a-z]*i)")

ev = json.load(sys.stdin)
if ev.get("agent_type") == "researcher":
    sys.exit(0)
tool, inp = ev.get("tool_name"), ev.get("tool_input") or {}
blocked = False
if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
    blocked = str(inp.get("file_path", "")).endswith(PROTECTED)
elif tool == "Bash":
    cmd = inp.get("command", "")
    blocked = PROTECTED in cmd and bool(WRITES.search(cmd)) and "git checkout" not in cmd
if blocked:
    print("Blocked: only the `researcher` subagent may change train.py. Call it with the Agent tool "
          "(subagent_type: researcher) and describe what you need.", file=sys.stderr)
    sys.exit(2)
sys.exit(0)
