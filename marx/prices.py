"""USD per million tokens. Sources: Anthropic pricing (Oct 2026, incl. Haiku 5.5) and DeepSeek's
pricing page (V4.1, Oct 2026; peak-hour rates used to be conservative).

cache_write_5m / cache_write_1h: Anthropic charges 1.25x / 2x input to write the
prompt cache; DeepSeek caches automatically with no write premium.
"""
import fnmatch

PRICES = {
    "claude-opus-5-5":   dict(input=4.00, output=20.00, cache_read=0.20, cache_write_5m=5.00, cache_write_1h=8.00),
    "claude-sonnet-5-5": dict(input=2.00, output=10.00, cache_read=0.20, cache_write_5m=2.50, cache_write_1h=4.00),
    "claude-haiku-4-5*": dict(input=1.00, output=5.00, cache_read=0.10, cache_write_5m=1.25, cache_write_1h=2.00),
    # Haiku 5.5 is priced by prompt length: prompts over 100k tokens pay 5x (see TIERS)
    "claude-haiku-5-5*": dict(input=0.10, output=0.50, cache_read=0.01, cache_write_5m=0.125, cache_write_1h=0.20),
    "deepseek-flash":    dict(input=0.30, output=1.20, cache_read=0.006, cache_write_5m=0.30, cache_write_1h=0.30),
    "deepseek-v4-pro":   dict(input=1.32, output=3.96, cache_read=0.044, cache_write_5m=1.32, cache_write_1h=1.32),
}


TIERS = {"claude-haiku-5-5*": (100_000, 5.0)}   # (prompt-token threshold, multiplier above it)


def price_for(model):
    for pattern, p in PRICES.items():
        if fnmatch.fnmatch(model, pattern):
            return p
    raise KeyError(f"no price for model {model!r}")


def usage_cost(model, usage):
    """Cost breakdown (USD) of one API call's `usage` block (Anthropic Messages format)."""
    p = price_for(model)
    prompt = sum(usage.get(k, 0) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    for pattern, (threshold, mult) in TIERS.items():
        if fnmatch.fnmatch(model, pattern) and prompt > threshold:
            p = {k: v * mult for k, v in p.items()}
    cc = usage.get("cache_creation") or {}
    w1h = cc.get("ephemeral_1h_input_tokens", 0) or 0
    w5m = cc.get("ephemeral_5m_input_tokens", 0) or 0
    if not cc:  # older/compatible APIs report only the total
        w5m = usage.get("cache_creation_input_tokens", 0) or 0
    parts = {
        "input": (usage.get("input_tokens", 0) or 0) * p["input"],
        "cache_write": w5m * p["cache_write_5m"] + w1h * p["cache_write_1h"],
        "cache_read": (usage.get("cache_read_input_tokens", 0) or 0) * p["cache_read"],
        "output": (usage.get("output_tokens", 0) or 0) * p["output"],
    }
    return {k: v / 1e6 for k, v in parts.items()}
