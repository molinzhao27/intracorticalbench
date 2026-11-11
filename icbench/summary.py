"""Compile the headline results into one chart-ready file, `results/summary.json`.

Everything is read from the saved cells in `runs/results`; nothing is typed in by hand. The prose
version is `results/SUMMARY.md`.

  uv run python -m icbench.summary
"""
from __future__ import annotations

import json

from .models import ROOT
from .report import paired
from .social import BOARD, EFFORTS, MODEL_ROWS, board_points

RESULTS = ROOT / "runs/results"
K = [0, 16, 64, 192]


def load(name: str) -> dict | None:
    f = RESULTS / f"sentences__{name}.json"
    c = json.loads(f.read_text()) if f.exists() else None
    return c if c and c.get("score") else None


def row(c: dict | None) -> dict | None:
    """The numbers a chart or a sentence needs from one cell."""
    if not c:
        return None
    s, u = c["score"], c.get("usage") or {}
    out = dict(wer=round(s["wer"], 3), chance=round(s["wer_chance_len"], 3), p=round(s["wer_p_len"], 4), above_chance=s["wer_p_len"] <= 0.01,
               exact=round(s["exact"], 3), answered=c["n"] - c.get("missing", 0), n=c["n"])
    if u.get("output"):
        out |= dict(output_tokens=u["output"], reasoning_tokens=u.get("reasoning", 0))
    return out


def main() -> None:
    # 1. In the prompt through the APIs: 8 held-out trials per request, labelled trials 0 -> 192.
    curve = {}
    for label, key in [("GPT Astra", "astra"), ("Claude Fable 5.2", "fable"), ("Claude Opus 5.5", "opus"), ("Claude Sonnet 5.5", "sonnet"), ("Claude Haiku 5.5", "haiku")]:
        curve[label] = {}
        for e in ("low", "high"):
            pts = []
            for k in K:
                r = row(load(f"t15_small__text__{key}__k{k}__{e}"))
                if r is None:
                    pts.append(dict(k=k, wer=None, note="no answer (output limit)"))
                    continue
                w = load(f"t15_small__none__{key}__k{k}__{e}")
                pts.append(dict(k=k, **r, withheld_wer=round(w["score"]["wer"], 3) if w else None))
            curve[label][e] = pts

    # 2. Reasoning sweep and zero-shot: one request with all 48 held-out trials, no tools.
    cost = {k: {key: {p[0]: round(p[1], 3) for p in pts} for key, pts in board_points(k).items()} for k in (192, 0)}
    sweep, zero = {}, {}
    for label, key, _ in BOARD:
        for k, dest in ((192, sweep), (0, zero)):
            dest[label] = {}
            for e in EFFORTS:
                r = row(load(f"t15_small__text__{key}__k{k}__{e}"))
                if r:
                    dest[label][e] = r | dict(cost_usd=cost[k].get(key, {}).get(e))

    # 3. The decoding method given as a hint: same cell with and without.
    hint = []
    for f in sorted(RESULTS.glob("sentences__t15_small__text__*-notools-hint__k192__*.json")):
        with_ = json.loads(f.read_text())
        base_key = with_["model"].removesuffix("-hint").replace("gpt-6-astra-codex", "astra-codex")
        without = load(f"t15_small__text__{base_key}__k192__{with_['effort']}")
        if not without:
            continue
        d, lo, hi, _ = paired(with_, without)
        label = next(n for n, key, _ in BOARD if key == base_key)
        hint.append(dict(model=label, effort=with_["effort"], without=row(without), with_hint=row(with_), change=round(d, 3), ci95=[round(lo, 3), round(hi, 3)]))
    hint.sort(key=lambda h: h["with_hint"]["wer"])

    # 4. With tools (the model writes and trains its own decoder), for contrast.
    tools = {label: row(load(f"t15_small__text__{agent}__k192__high")) for label, _, agent, _ in MODEL_ROWS}
    tools = {k: v for k, v in tools.items() if v}

    # 5. Seven isolated words, and classical decoders on the same input.
    report = json.loads((ROOT / "results/report.json").read_text())
    base = json.loads((ROOT / "results/baselines.json").read_text())
    words = [dict(participant=w["dataset"], model=w["model"], effort=w["effort"], labelled=w["k"], correct=round(w["acc"] * w["n"]), n=w["n"]) for w in report["words"]]

    # 6. A few of the best cell's sentences, right and wrong.
    best = load("t15_small__text__astra__k192__high")
    examples = [dict(said=r, decoded=p) for p, r in zip(best["preds"], best["refs"])][:12]

    out = dict(
        about=dict(
            data="t15_small: T15, 2023.08.11, 50-word vocabulary, Card et al. 2024 (Brain-to-Text '25). 48 held-out sentences, one random split.",
            metric="Word error rate (lower is better). `chance` is the duration-matched permutation null for that cell; above_chance means p <= 0.01.",
            routes="in_prompt_curve: provider APIs, 8 held-out trials per request. sweep / zero_shot / hint: one request with all 48 held-out trials.",
            cost="cost_usd is tokens at list price for one pass over the prompt, not what was paid."),
        in_prompt_curve=curve, sweep=sweep, zero_shot=zero, hint=hint, tools=tools,
        seven_words=dict(models=words, classical=base["words"]),
        classical_sentences=[dict(decoder=b["decoder"], k=b["k"], wer=round(b["score"]["wer"], 3), chance=round(b["score"]["wer_chance_len"], 3),
                                  p=round(b["score"]["wer_p_len"], 4)) for b in base["sentences"] if b["dataset"] == "t15_small"],
        examples=examples)
    (ROOT / "results/summary.json").write_text(json.dumps(out, indent=1))
    n_zero = sum(len(v) for v in zero.values())
    print(f"wrote results/summary.json: sweep {sum(len(v) for v in sweep.values())} cells, zero-shot {n_zero}, hint {len(hint)}, tools {len(tools)}")
    for h in hint:
        print(f"  hint {h['model']:20s} {h['effort']:6s} {h['without']['wer']:.2f} -> {h['with_hint']['wer']:.2f}")


if __name__ == "__main__":
    main()
