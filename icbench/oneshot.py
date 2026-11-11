"""The no-tools test as one plain chat request to any OpenAI-compatible API (used for DeepSeek).

The request is the same text the Codex and Claude Code no-tools runs receive: 192 labelled trials
and all 48 held-out trials. Nothing but the prompt goes in, nothing but text comes out.

  uv run python -m icbench.oneshot deepseek-flash low --max-tokens 12000

Spending is guarded by the provider's own balance endpoint: the call is skipped unless the balance
covers its worst case (every prompt token uncached, output up to max_tokens) plus a reserve.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from . import agentic
from .models import ROOT

BASE = "https://api.deepseek.com"
# USD per million tokens at peak hours (off-peak is half): uncached input, output
PEAK = {"deepseek-flash": (0.30, 1.20), "deepseek-v4-pro": (1.32, 3.96)}
SYSTEM = "You are a careful analyst. Follow the instructions in the user message exactly."


def balance() -> float:
    req = urllib.request.Request(BASE + "/user/balance", headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"], "Accept": "application/json"})
    info = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return float(next(b["total_balance"] for b in info["balance_infos"] if b["currency"] == "USD"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("effort", choices=["none", "low", "high", "max"])
    p.add_argument("--dataset", default="t15_small")
    p.add_argument("--k", type=int, default=192)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--reserve", type=float, default=0.03, help="USD that must remain after the worst case of this call")
    p.add_argument("--prompt-file", type=Path, help="use this prompt text instead of the standard one (same trials)")
    p.add_argument("--tag", default="", help="suffix for the run folder and the model label, e.g. hint")
    a = p.parse_args()
    import openai
    out = ROOT / f"runs/oneshot/{a.model}_{a.dataset}_k{a.k}_{a.effort}{'_' + a.tag if a.tag else ''}"
    agentic.make_text(a.dataset, out, a.k, agentic._args())
    if a.prompt_file:
        (out / "prompt.txt").write_text(a.prompt_file.read_text())
    prompt = (out / "prompt.txt").read_text()
    pin, pout = PEAK[a.model]
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)                # peak: 01-04 and 06-10 UTC on weekdays; otherwise half price
    peak = now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)
    worst = (len(prompt) / 1.7 * pin + a.max_tokens * pout) / 1e6 * (1.0 if peak else 0.5)   # nothing cached, output to the limit
    before = balance()
    if before - worst < a.reserve:
        print(f"SKIPPED {a.model} {a.effort}: balance ${before:.2f}, worst case ${worst:.2f}")
        return
    client = openai.OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=BASE, timeout=3600, max_retries=2)
    kw = dict(extra_body={"thinking": {"type": "disabled"}}) if a.effort == "none" else dict(reasoning_effort=a.effort)
    text, usage = "", None
    stream = client.chat.completions.create(model=a.model, max_tokens=a.max_tokens, stream=True, stream_options={"include_usage": True},
                                            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}], **kw)
    finish = None
    for chunk in stream:
        if chunk.usage:
            usage = chunk.usage.model_dump()
        for ch in chunk.choices:
            text += ch.delta.content or ""
            finish = ch.finish_reason or finish
    (out / "answer.txt").write_text(text)
    details = (usage or {}).get("completion_tokens_details") or {}
    log = dict(type="turn.completed", finish=finish, raw=usage,
               usage=dict(input_tokens=(usage or {}).get("prompt_tokens", 0), cached_input_tokens=(usage or {}).get("prompt_cache_hit_tokens", 0),
                          output_tokens=(usage or {}).get("completion_tokens", 0), reasoning_output_tokens=details.get("reasoning_tokens", 0)))
    (out / "run.log").write_text(json.dumps(log) + "\n")
    after = balance()
    print(f"{a.model} {a.effort}: finish={finish} prompt={log['usage']['input_tokens']} (cached {log['usage']['cached_input_tokens']}) "
          f"output={log['usage']['output_tokens']} (reasoning {log['usage']['reasoning_output_tokens']}) | balance ${before:.2f} -> ${after:.2f}")
    if text.strip():
        agentic.score_images(a.dataset, out, a.k, f"{a.model}-api-notools{'-' + a.tag if a.tag else ''}", a.effort, out / "run.log", agentic._args(), "text")
    else:
        print("no answer text (output limit reached before the answer?)")


if __name__ == "__main__":
    main()
