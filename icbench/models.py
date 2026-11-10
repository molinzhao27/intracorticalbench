"""Provider calls with a disk cache (re-runs are free) and a hard spending cap per provider.

Calls are plain single-turn requests: no tools, no code execution, no model fallback. The model
only sees the prompt, so anything it decodes it decodes itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
CACHE = ROOT / "runs/cache"
LEDGER = ROOT / "runs/ledger.json"
_lock = threading.Lock()

# USD per million tokens: input, output, cache read, cache write multiplier on input
MODELS = {
    "fable": dict(provider="anthropic", id="claude-fable-5-2", inp=8.0, out=40.0, read=0.20, write=1.25),
    "opus": dict(provider="anthropic", id="claude-opus-5-5", inp=4.0, out=20.0, read=0.20, write=1.25),
    # OpenAI pricing page, standard tier; every rate doubles (output x1.5) above 272k input tokens.
    "astra": dict(provider="openai", id="gpt-6-astra", inp=10.0, out=50.0, read=1.0, write=1.25),
}
CAPS = {"anthropic": float(os.environ.get("ICB_CAP_ANTHROPIC", 185)), "openai": float(os.environ.get("ICB_CAP_OPENAI", 88))}


BUDGET = 40000   # output tokens per call the Claude models are told they have
OPENAI_MAX_OUT = 60000   # hard output limit per OpenAI call (reasoning included)


class BudgetExceeded(RuntimeError):
    pass


def spent() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {"anthropic": 0.0, "openai": 0.0}


def _charge(provider: str, usd: float) -> None:
    with _lock:
        s = spent()
        s[provider] = s.get(provider, 0.0) + usd
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        LEDGER.write_text(json.dumps(s, indent=1))


def call(model: str, effort: str, system: str, blocks: list[dict], tag: str = "") -> dict:
    """blocks: {"type": "text", "text"} | {"type": "image", "b64"} | {"type": "cache"} (prefix marker)."""
    m = MODELS[model]
    key = hashlib.sha256(json.dumps([m["id"], effort, system, blocks], sort_keys=True).encode()).hexdigest()[:24]
    path = CACHE / f"{key}.json"
    if path.exists():
        old = json.loads(path.read_text())
        # A call that ran out of output tokens or was declined is retried once, not reused.
        if old["stop"] not in ("max_tokens", "refusal", "incomplete") or old.get("retried") or os.environ.get("ICB_OFFLINE"):
            return old | {"cached_locally": True}
        retried = True
    else:
        retried = False
    if os.environ.get("ICB_OFFLINE"):                             # score what is cached, call nothing
        return dict(text="", stop="not_run", cost=0.0, model=m["id"], usage=dict(input=0, cache_write=0, cache_read=0, output=0))
    if (ROOT / "runs" / f"FROZEN_{m['provider']}").exists():        # user paused this provider; delete the file to resume
        raise BudgetExceeded(f"{m['provider']} runs are paused until the user approves more spend (runs/FROZEN_{m['provider']})")
    if spent().get(m["provider"], 0.0) >= CAPS[m["provider"]]:
        raise BudgetExceeded(f"{m['provider']} cap of ${CAPS[m['provider']]:.0f} reached")
    out = (_anthropic if m["provider"] == "anthropic" else _openai)(m, effort, system, blocks)
    out["tag"], out["retried"] = tag, retried
    _charge(m["provider"], out["cost"])
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def _anthropic(m: dict, effort: str, system: str, blocks: list[dict]) -> dict:
    import anthropic
    client = anthropic.Anthropic(max_retries=6, timeout=3600)
    content: list[dict] = []
    for b in blocks:
        if b["type"] == "text":
            content.append({"type": "text", "text": b["text"]})
        elif b["type"] == "image":
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b["b64"]}})
        elif content:                                             # cache marker on the block before it
            content[-1]["cache_control"] = {"type": "ephemeral"}
    # Thinking is always on for these models; depth is set by effort. No `fallbacks`: a refusal
    # must not be silently answered by a different model in a benchmark.
    # The task budget tells the model how many tokens (thinking included) it has for the reply, so
    # that at high effort it paces itself and answers instead of thinking until max_tokens.
    with client.beta.messages.stream(model=m["id"], max_tokens=64000, system=system,
                                     messages=[{"role": "user", "content": content}],
                                     betas=["task-budgets-2026-03-13"],
                                     output_config={"effort": effort, "task_budget": {"type": "tokens", "total": BUDGET}}) as s:
        msg = s.get_final_message()
    u = msg.usage
    w, r = u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0
    cost = (u.input_tokens * m["inp"] + w * m["inp"] * m["write"] + r * m["read"] + u.output_tokens * m["out"]) / 1e6
    text = "".join(b.text for b in msg.content if b.type == "text")
    details = getattr(msg, "stop_details", None)
    return dict(text=text, stop=msg.stop_reason, cost=cost, model=msg.model,
                refusal=getattr(details, "category", None) if msg.stop_reason == "refusal" else None,
                usage=dict(input=u.input_tokens, cache_write=w, cache_read=r, output=u.output_tokens))


def _openai(m: dict, effort: str, system: str, blocks: list[dict]) -> dict:
    import openai
    client = openai.OpenAI(max_retries=6, timeout=3600)
    content = []
    for b in blocks:
        if b["type"] == "text":
            content.append({"type": "input_text", "text": b["text"]})
        elif b["type"] == "image":
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b['b64']}"})
        elif content:
            # On this model family a cache hit needs an exact match to a written breakpoint, and the
            # automatic one sits after the (changing) queries. Mark the end of the labelled trials.
            content[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    with client.responses.stream(model=m["id"], instructions=system, input=[{"role": "user", "content": content}],
                                 reasoning={"effort": effort}, max_output_tokens=OPENAI_MAX_OUT,
                                 extra_body={"prompt_cache_options": {"mode": "explicit"}}) as s:
        resp = None
        for ev in s:                                              # keep the final object even when the
            if ev.type in ("response.completed", "response.incomplete", "response.failed"):   # reply is cut off
                resp = ev.response
    u = resp.usage
    cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
    wrote = getattr(u.input_tokens_details, "cache_write_tokens", 0) or 0
    reasoning = getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0
    long, out = (2.0, 75.0) if u.input_tokens > 272_000 else (1.0, m["out"])
    plain = u.input_tokens - cached - wrote
    cost = (long * (plain * m["inp"] + wrote * m["inp"] * m["write"] + cached * m["read"]) + u.output_tokens * out) / 1e6
    return dict(text=resp.output_text, stop=resp.status, cost=cost, model=resp.model,
                usage=dict(input=plain, cache_write=wrote, cache_read=cached,
                           output=u.output_tokens, reasoning=reasoning))
