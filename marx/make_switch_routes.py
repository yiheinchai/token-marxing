"""Generate the same-conversation switching route configs (marx/routes.switch*.json).

    python -m marx.make_switch_routes

The labour mode's system text is claude/OPERATOR.switch.md; edit that file and re-run this.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

THINKER = ("You are the THINKER (expensive model) of this two-model session. Do the research thinking and "
           "write train.py; never run check.py, jobq.py, sleep or read job logs yourself. When the code is "
           "ready, run `marx-mode labour \"<exp N, name, one-line desc>\"` in the same message as your last edit. "
           "Assistant turns after a `marx-mode labour` hand-off, up to `marx-mode think`, were written by the "
           "OPERATOR (a different model), not by you.")
TURN_NOTE = ("You are now the OPERATOR (cheap model); the THINKER handed the conversation to you. Act: do the "
             "mechanical pipeline work with your tools, sleeping before every status poll. When you are done, "
             "run `marx-mode think \"<short report>\"` and stop right after it; never end with plain text.")
# What the cheap upstream needs changed in Claude Code's Opus-shaped requests.
CHEAP_COMPAT = {"drop_fields": ["thinking", "output_config", "context_management"], "strip_thinking": True,
                "fold_system_messages": True, "max_tokens": 32000}
HANDOFF_NOTE = ("The THINKER (a different model) wrote every assistant turn above this point, including any edits "
                "to train.py: that code is the experiment to run, not your work, so do not revert it. You are now "
                "the OPERATOR: do the mechanical pipeline work with your tools, sleeping before every status poll, "
                "then run `marx-mode think \"<short report>\"` and stop right after it; never end with plain text.")
CHEAP = {
    # Haiku 4.5 (used in the first experiments): needs request rewrites; per-turn role note
    "anthropic": dict({"upstream": "https://api.anthropic.com", "model": "claude-haiku-4-5-20251001"},
                      **CHEAP_COMPAT, turn_note=TURN_NOTE),
    # Haiku 5.5 takes Claude Code's Opus-shaped requests as they are; the role note is pinned to
    # each hand-off so the history it sees never changes (it has preserved-thinking checks too)
    "haiku55": {"upstream": "https://api.anthropic.com", "model": "claude-haiku-5-5", "handoff_note": HANDOFF_NOTE},
    "deepseek": dict({"upstream": "https://api.deepseek.com/anthropic", "api_key_env": "DEEPSEEK_API_KEY",
                      "model": "deepseek-flash"}, **CHEAP_COMPAT, handoff_note=HANDOFF_NOTE),
}


def config(provider, elide):
    operator = open(os.path.join(ROOT, "claude", "OPERATOR.switch.md")).read().strip()
    think = {"upstream": "https://api.anthropic.com", "system_append": THINKER}
    if elide:
        think["elide_other_modes"] = True
    labour = dict(CHEAP[provider], system_append=operator)
    return {"routes": [
        {"name": "main", "match": "claude-opus-*",
         "switch": {"default": "think", "modes": {"think": think, "labour": labour}}},
        {"name": "other", "match": "*", "upstream": "https://api.anthropic.com"},
    ]}


def main():
    for provider in CHEAP:
        for elide in (False, True):
            name = f"routes.switch{'_elide' if elide else ''}.{provider}.json"
            with open(os.path.join(HERE, name), "w") as fh:
                json.dump(config(provider, elide), fh, indent=2)
                fh.write("\n")
            print(name)


if __name__ == "__main__":
    main()
