#!/usr/bin/env python3
"""Stop hook for same-conversation model switching (marx-mode).

If the agent tries to end its turn while the conversation is in labour mode, the cheap
operator model is about to drop the baton (observed: "I need to wait for the THINKER..."),
which in a headless run ends the whole session. Block the stop and tell it to hand back.
"""
import json
import os
import sys

ev = json.load(sys.stdin)
path = os.environ.get("MARX_MODE_FILE") or os.path.join(ev.get("cwd") or ".", ".marx-mode")
try:
    mode = open(path).read().strip()
except OSError:
    mode = ""
if mode == "labour":
    print(json.dumps({"decision": "block", "reason": (
        "You are the OPERATOR and the conversation is still in labour mode. Finish the pipeline "
        "work if any is left, then hand back with `marx-mode think \"<short report>\"`. "
        "Do not end your turn with plain text.")}))
sys.exit(0)
