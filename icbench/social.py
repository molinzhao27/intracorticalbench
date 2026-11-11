"""Static PNG figures (1600 x 900) for posts and the README.

  uv run python -m icbench.social        # writes assets/figures/*.png
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import ROOT
from .run import LETTERS, prepare

OUT = ROOT / "assets/figures"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # validated categorical slots 1-3
RAMP = ["#86b6ef", "#2a78d6", "#104281"]                        # one hue, light to dark, for ordered conditions
GOOD, BAD = "#006300", "#d03b3b"
plt.rcParams.update({"font.family": ["Avenir Next", "Helvetica Neue", "DejaVu Sans"], "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def cell(name: str) -> dict:
    return json.loads((ROOT / f"runs/results/sentences__{name}.json").read_text())


def frame(fig, title: str, sub: str, foot: str) -> None:
    import textwrap
    fig.text(0.05, 0.935, title, fontsize=17, fontweight="bold", color=INK, va="center")
    fig.text(0.05, 0.89, textwrap.fill(sub, 100), fontsize=10, color=INK2, va="top", linespacing=1.35)
    fig.text(0.05, 0.075, textwrap.fill(foot, 132), fontsize=7.8, color=MUTED, va="top", linespacing=1.35)


def r2(x: float) -> str:
    return f"{x + 1e-9:.2f}"                                      # round half up, as in the write-up


def learning_curve() -> None:
    ks = [0, 16, 64, 192]
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    ax = fig.add_axes([0.085, 0.215, 0.58, 0.535])
    frame(fig, "Zero-shot fails. With examples in the prompt, it decodes.",
          "Word error rate on held-out sentences. Attempted speech recorded with intracortical arrays, 50-word vocabulary. Plain model calls: no tools, no training.",
          "48 held-out sentences per point. Guessing = the same prompts with the neural activity removed. "
          "† no answer within the output limit at 192. Data: Card et al. 2024 (participant T15).")
    series = [("opus", "high", AQUA, "-", "Opus 5.5, high effort"), ("fable", "low", ORANGE, "--", "Fable 5.2, low effort"),
              ("fable", "high", ORANGE, "-", "Fable 5.2, high effort"), ("astra", "low", BLUE, "--", "GPT Astra, low effort"),
              ("astra", "high", BLUE, "-", "GPT Astra, high effort")]
    ends = []
    guess = []
    for k in ks:
        c = cell(f"t15_small__none__astra__k{k}__low")
        guess.append(c["score"]["wer"])
    ax.plot(range(4), guess, color=MUTED, lw=1.5, zorder=1)
    ax.text(1.5, guess[1] + 0.045, "guessing without the neural activity", fontsize=8.5, color=MUTED, ha="center")
    for model, eff, col, ls, label in series:
        xs, ys = [], []
        for i, k in enumerate(ks):
            p = ROOT / f"runs/results/sentences__t15_small__text__{model}__k{k}__{eff}.json"
            if p.exists() and json.loads(p.read_text())["score"]:
                xs.append(i)
                ys.append(json.loads(p.read_text())["score"]["wer"])
        ax.plot(xs, ys, color=col, lw=2, ls=ls, marker="o", ms=6.5, mfc=col if ls == "-" else SURFACE, mec=col, mew=1.8, zorder=3)
        ends.append((ys[-1], xs[-1], label, col))
    ly = None
    for y, x, label, col in sorted(ends, key=lambda e: -e[0]):   # direct labels, pushed apart
        yy = y if ly is None else min(y, ly - 0.075)
        ly = yy
        ax.plot([x + 0.06, 3.13], [y, yy], color=GRID, lw=0.8, clip_on=False, zorder=0)
        ax.scatter([3.19], [yy], s=42, marker="s", color=col, clip_on=False, zorder=4)
        ax.text(3.27, yy, label + ("" if x == 3 else " †"), fontsize=9, color=INK, va="center", clip_on=False)
    best = cell("t15_small__text__astra__k192__high")["score"]
    ax.annotate(f"{best['exact'] * 100:.0f}% of sentences\nexactly right", xy=(3, best["wer"]), xytext=(1.55, 0.05), fontsize=9, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=5))
    ax.set_xlim(-0.15, 3.1); ax.set_ylim(0, 1.32)
    ax.set_xticks(range(4)); ax.set_xticklabels(["0\n(zero-shot)", "16", "64", "192"], fontsize=9.5, color=INK2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]); ax.tick_params(axis="y", labelsize=9, colors=MUTED, length=0); ax.tick_params(axis="x", length=0)
    ax.set_xlabel("labelled trials (neural activity + sentence) placed in the prompt", fontsize=9.5, color=INK2, labelpad=6)
    ax.set_ylabel("word error rate", fontsize=9.5, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(OUT / "1_in_prompt.png")
    plt.close(fig)


def ladder() -> None:
    conds = ["In the prompt, no tools", "Code to look at a 12-number summary", "Code to manipulate the full recording, no ML", "Free to write and train a decoder"]
    ramp = ["#b7d3f6", "#6da7ec", "#2a78d6", "#104281"]              # one hue, light to dark, in order of freedom
    # in-prompt cells at high effort, like the tool runs: API for the first, Codex with tools forbidden for the others
    rows = [("50-word vocabulary, 192 examples", "t15_small", 192, "t15_small__text__astra__k192__high"),
            ("Open vocabulary, 192 examples", "t15_open", 192, "t15_open__text__astra-codex-notools__k192__high"),
            ("Second participant, 72 examples", "t16_small", 72, "t16_small__text__astra-codex-notools__k72__high")]
    fig = plt.figure(figsize=(8, 5.6), dpi=200)
    import textwrap
    fig.text(0.05, 0.95, "Small vocabulary: the prompt is enough. Open vocabulary needs tools.", fontsize=12.8, fontweight="bold", color=INK, va="center")
    fig.text(0.05, 0.915, textwrap.fill("GPT Astra: word error rate on 48 held-out sentences, with labelled trials from the same session. "
                                         "Grey tick = chance. No ML = nothing fitted or trained; the model compares segments itself.", 100),
             fontsize=10, color=INK2, va="top", linespacing=1.35)
    fig.text(0.05, 0.05, textwrap.fill("All bars: high reasoning effort, one run each. Chance = the same predictions re-paired with other trials of matching duration. "
                                         "Tool runs: Codex CLI, sandboxed and audited. Data: Card et al. 2024 (first two panels), Kunz et al. 2025 (third).", 132),
             fontsize=7.8, color=MUTED, va="top", linespacing=1.35)
    for j, (name, ds, k, prompt_cell) in enumerate(rows):
        ax = fig.add_axes([0.40, 0.615 - j * 0.235, 0.53, 0.175])
        cells = [prompt_cell, f"{ds}__text__astra-codex-lite__k{k}__high",
                 f"{ds}__text__astra-codex-manipulate__k{k}__high", f"{ds}__text__astra-codex__k{k}__high"]
        for i, c in enumerate(cells):
            sc = cell(c)["score"]
            y = 3 - i
            ax.barh(y, sc["wer"], height=0.66, color=ramp[i], zorder=2)
            ax.plot([sc["wer_chance_len"]] * 2, [y - 0.4, y + 0.4], color=MUTED, lw=2, zorder=3)
            near = abs(sc["wer"] - sc["wer_chance_len"]) < 0.09
            ax.text((sc["wer_chance_len"] if near else sc["wer"]) + 0.02, y, r2(sc["wer"]), fontsize=9.5, fontweight="bold", color=INK, va="center")
            ax.text(-0.03, y, conds[i], fontsize=8.8, color=INK, va="center", ha="right")
        ax.text(-0.03, 4.02, name, fontsize=10, fontweight="bold", color=INK, ha="right", va="center")
        ax.set_xlim(0, 1.25); ax.set_ylim(-0.55, 3.6)
        ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0, labelbottom=(j == 2))
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if j == 2:
            ax.set_xlabel("word error rate (lower is better)", fontsize=9.5, color=INK2, labelpad=3)
    fig.savefig(OUT / "2_tools_ladder.png")
    plt.close(fig)


def examples() -> None:
    a = SimpleNamespace(seed=0, nq=48, kmax=192, pcs=12, nq_per_word=6)
    queries, ctx, _, _, _ = prepare(a, "sentences", "t15_small", 192)
    res = cell("t15_small__text__astra__k192__high")
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    frame(fig, "What goes in, what comes out.",
          "GPT Astra, high effort, no tools. 192 labelled trials in the prompt, then held-out trials.",
          "First 14 of 48 held-out sentences (T15, 50 words; Card et al. 2024). Each letter: one component's decile in a 160 ms frame.")
    fig.text(0.05, 0.775, "IN: neural activity, as text", fontsize=9.5, fontweight="bold", color=INK2)
    q, label, sec = ctx[0]
    lines = [" ".join(LETTERS[v] for v in row) for row in q[:13]]
    fig.text(0.05, 0.74, f"Trial 1 ({sec:.1f} s)\nSentence: {label}\n" + "\n".join(lines) + "\n...", fontsize=7.6, family="DejaVu Sans Mono", color=INK, va="top", linespacing=1.38)
    fig.text(0.05, 0.155, f"{len(q)} frames for this trial; 192 labelled trials,\nthen the held-out trials with no sentence", fontsize=8.5, color=MUTED, va="center")
    fig.text(0.415, 0.775, "Attempted sentence", fontsize=9.5, fontweight="bold", color=INK2)
    fig.text(0.69, 0.775, "OUT: decoded by the model", fontsize=9.5, fontweight="bold", color=INK2)
    y = 0.725
    for p, t in list(zip(res["preds"], res["refs"]))[:14]:
        ok = data_norm(p) == data_norm(t)
        fig.text(0.415, y, t, fontsize=9, color=INK, va="center")
        fig.text(0.69, y, p, fontsize=9, color=INK if ok else BAD, va="center")
        fig.text(0.955, y, "✓" if ok else "✗", fontsize=10, color=GOOD if ok else BAD, va="center", ha="center", fontweight="bold")
        y -= 0.0425
    fig.add_artist(plt.Line2D([0.39, 0.39], [0.12, 0.80], color=GRID, lw=1))
    fig.savefig(OUT / "3_examples.png")
    plt.close(fig)


MODEL_ROWS = [("GPT Astra", "astra", "astra-codex", BLUE), ("gpt-6.1-sol", "gpt-6.1-sol-codex-notools", "gpt-6.1-sol-codex", BLUE),
              ("gpt-6.1-terra", "gpt-6.1-terra-codex-notools", "gpt-6.1-terra-codex", BLUE), ("gpt-6-sol", "gpt-6-sol-codex-notools", "gpt-6-sol-codex", BLUE),
              ("gpt-6-luna", "gpt-6-luna-codex-notools", "gpt-6-luna-codex", BLUE), ("gpt-5.6-sol", "gpt-5.6-sol-codex-notools", "gpt-5.6-sol-codex", BLUE),
              ("gpt-5.6-terra", "gpt-5.6-terra-codex-notools", "gpt-5.6-terra-codex", BLUE), ("gpt-5.6-luna", "gpt-5.6-luna-codex-notools", "gpt-5.6-luna-codex", BLUE),
              ("Claude Fable 5.2", "fable", "fable-claudecode", ORANGE), ("Claude Opus 5.5", "opus", "opus-claudecode", ORANGE),
              ("Claude Sonnet 5.5", "sonnet", "sonnet-claudecode", ORANGE), ("Claude Haiku 5.5", "haiku", "haiku-claudecode", ORANGE),
              ("DeepSeek V4.1 Flash", "deepseek-flash-api-notools", "deepseek-flash-agent", AQUA),
              ("DeepSeek V4 Pro", "deepseek-v4-pro-api-notools", "deepseek-v4-pro-agent", AQUA)]


def _score(name: str):
    f = ROOT / f"runs/results/sentences__{name}.json"
    c = json.loads(f.read_text()) if f.exists() else None
    return c["score"] if c and c.get("score") else None


def _model_chart(fname: str, title: str, sub: str, foot: str, rows: list, legend: list) -> None:
    """One row per model, sorted by its best score. rows: (label, colour, [(score, style)], chance, significant)."""
    rows = sorted(rows, key=lambda r: min(s["wer"] for s, _ in r[2]))
    n = len(rows)
    fig = plt.figure(figsize=(8, 0.9 + 0.36 * n + 0.75), dpi=200)
    h = fig.get_figheight()
    fig.text(0.05, 1 - 0.36 / h, title, fontsize=16, fontweight="bold", color=INK, va="center")
    fig.text(0.05, 1 - 0.68 / h, sub, fontsize=10.5, color=INK2, va="center")
    fig.text(0.05, 0.17 / h, foot, fontsize=8, color=MUTED, va="center")
    ax = fig.add_axes([0.25, 0.72 / h, 0.58, 0.36 * n / h])
    for i, (label, col, pts, chance, sig) in enumerate(rows):
        y = n - 1 - i
        ws = [s["wer"] for s, _ in pts]
        ax.plot([min(ws), max(ws)], [y, y], color=col, lw=2, zorder=2, solid_capstyle="round", alpha=0.55)
        for s_, style in pts:
            ax.plot(s_["wer"], y, "o", ms=7.5, mfc=col if style == "filled" else SURFACE, mec=col, mew=1.8, zorder=3 if style == "open" else 4)
        ax.plot([chance] * 2, [y - 0.33, y + 0.33], color=MUTED, lw=2, zorder=1)
        ax.text(-0.03, y, label, fontsize=10, color=INK, va="center", ha="right")
        ax.text(1.32, y, r2(min(ws)) if sig else "chance", fontsize=10, fontweight="bold" if sig else "normal", color=INK if sig else MUTED, va="center", ha="left")
    ax.text(1.32, n - 0.2, "best", fontsize=8.5, color=MUTED, ha="left", va="center")
    ax.set_xlim(0, 1.28); ax.set_ylim(-0.7, n - 0.4)
    ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=9, colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("word error rate", fontsize=10, color=INK2, labelpad=3)
    for k, (text, col, style) in enumerate(legend):                # small key in the empty lower-left corner
        y = len(legend) - 1 - k - 0.15
        if style == "tick":
            ax.plot([0.045] * 2, [y - 0.3, y + 0.3], color=MUTED, lw=2)
        else:
            ax.plot(0.045, y, "o", ms=7, mfc=col if style == "filled" else SURFACE, mec=col, mew=1.8)
        ax.text(0.08, y, text, fontsize=9, color=INK2, va="center")
    fig.savefig(OUT / fname)
    plt.close(fig)


def models() -> None:
    """Every model tested without tools on the 50-word session at 192 labelled trials."""
    rows = []
    for label, m, _, col in MODEL_ROWS:
        sc = {e: _score(f"t15_small__text__{m}__k192__{e}") for e in ("low", "high")}
        pts = [(sc["low"], "open")] * bool(sc["low"]) + [(sc["high"], "filled")] * bool(sc["high"])
        vals = [s for s, _ in pts]
        rows.append((label, col, pts, sum(v["wer_chance_len"] for v in vals) / len(vals), min(v["wer_p_len"] for v in vals) <= 0.01))
    _model_chart("4_models.png", "No tools: one model decodes well, most stay at chance",
                 "Neural activity and 192 labelled examples in the prompt. 50-word vocabulary, 48 held-out sentences.",
                 "Blue: OpenAI. Orange: Anthropic. Green: DeepSeek. Fable gave no answer at high effort. Data: Card et al. 2024.", rows,
                 [("high reasoning effort", INK2, "filled"), ("low reasoning effort", INK2, "open"), ("chance", MUTED, "tick")])


def models_tools() -> None:
    """The same rows, for runs where the model may write and run code (one run each, high effort)."""
    rows = []
    for label, m, agent, col in MODEL_ROWS:
        full = _score(f"t15_small__text__{agent}__k192__high")
        if not full:
            continue
        prompt = [s for s in (_score(f"t15_small__text__{m}__k192__{e}") for e in ("low", "high")) if s]
        best = min(prompt, key=lambda s: s["wer"])
        rows.append((label, col, [(best, "open"), (full, "filled")], full["wer_chance_len"], full["wer_p_len"] <= 0.01))
    _model_chart("6_models_tools.png", "With tools: the model writes and trains its own decoder",
                 "Same data as files; the model may write and run any code. 48 held-out sentences, one run each.",
                 "Blue: OpenAI (Codex CLI). Orange: Anthropic (Claude Code). Sandboxed, no network. Data: Card et al. 2024.", rows,
                 [("with tools", INK2, "filled"), ("in the prompt, no tools", INK2, "open"), ("chance", MUTED, "tick")])


def tools_both() -> None:
    """GPT Astra and Fable 5.2 across the three levels of freedom, on three datasets."""
    import textwrap
    conds = ["In the prompt, no tools", "Code on the full recording, no fitting*", "Writes and trains a decoder"]
    panels = [("50-word vocabulary, 192 examples",
               [("t15_small__text__astra__k192__high", "t15_small__text__fable__k192__low"),
                ("t15_small__text__astra-codex-manipulate__k192__high", "t15_small__text__fable-claudecode-manipulate__k192__high"),
                ("t15_small__text__astra-codex__k192__high", "t15_small__text__fable-claudecode__k192__high")]),
              ("Open vocabulary, 192 examples",
               [("t15_open__text__astra-codex-notools__k192__high", "t15_open__text__fable__k192__high"),
                ("t15_open__text__astra-codex-manipulate__k192__high", "t15_open__text__fable-claudecode-manipulate__k192__high"),
                ("t15_open__text__astra-codex__k192__high", "t15_open__text__fable-claudecode__k192__high")]),
              ("Second participant, 72 examples",
               [("t16_small__text__astra-codex-notools__k72__high", "t16_small__text__fable__k72__high"),
                ("t16_small__text__astra-codex-manipulate__k72__high", "t16_small__text__fable-claudecode-manipulate__k72__high"),
                ("t16_small__text__astra-codex__k72__high", "t16_small__text__fable-claudecode__k72__high")])]
    fig = plt.figure(figsize=(8, 5.6), dpi=200)
    fig.text(0.05, 0.95, "With code, both models decode far better than in the prompt.", fontsize=14.5, fontweight="bold", color=INK, va="center")
    fig.text(0.05, 0.915, textwrap.fill("Word error rate on 48 held-out sentences, high reasoning effort, one run per point. Labelled trials from the same session. "
                                         "Grey tick = chance.", 100), fontsize=10, color=INK2, va="top", linespacing=1.35)
    fig.text(0.05, 0.062, textwrap.fill("* Nothing fitted or trained; the model types every sentence. Not like-for-like: Fable's code ranked each test segment against all labelled "
                                         "segments, GPT Astra compared hand-picked ones. Fable in the prompt at 192 examples is at low effort.", 132),
             fontsize=7.8, color=MUTED, va="top", linespacing=1.35)
    for j, (name, rows) in enumerate(panels):
        ax = fig.add_axes([0.40, 0.625 - j * 0.235, 0.53, 0.165])
        for i, (a, f) in enumerate(rows):
            y = 2 - i
            sa, sf = cell(a)["score"], cell(f)["score"]
            ax.plot([sa["wer"], sf["wer"]], [y, y], color=GRID, lw=2, zorder=1)
            ax.plot([(sa["wer_chance_len"] + sf["wer_chance_len"]) / 2] * 2, [y - 0.36, y + 0.36], color=MUTED, lw=2, zorder=1)
            for sc, col, dy in ((sa, BLUE, 0.0), (sf, ORANGE, 0.0)):
                ax.plot(sc["wer"], y, "o", ms=8, color=col, mec=SURFACE, mew=1.2, zorder=3)
            lo, hi = sorted([(sa["wer"], BLUE), (sf["wer"], ORANGE)])
            if hi[0] - lo[0] > 0.11:                              # far apart: a label over each point
                ax.text(lo[0], y + 0.3, r2(lo[0]), fontsize=8.5, fontweight="bold", color=INK, ha="center", va="bottom")
                ax.text(hi[0], y + 0.3, r2(hi[0]), fontsize=8.5, fontweight="bold", color=INK, ha="center", va="bottom")
            else:                                                 # close together: one label to the right, in point order
                ax.text(hi[0] + 0.035, y, f"{r2(lo[0])} / {r2(hi[0])}", fontsize=8.5, fontweight="bold", color=INK, ha="left", va="center")
            ax.text(-0.04, y, conds[i], fontsize=8.8, color=INK, va="center", ha="right")
        ax.text(-0.04, 3.45, name, fontsize=10, fontweight="bold", color=INK, ha="right", va="center")
        ax.set_xlim(0, 1.25); ax.set_ylim(-0.6, 3.0)
        ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0, labelbottom=(j == 2))
        ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
        if j == 2:
            ax.set_xlabel("word error rate (lower is better)", fontsize=9.5, color=INK2, labelpad=3)
        if j == 0:
            ax.plot(0.74, 3.45, "o", ms=8, color=BLUE, clip_on=False); ax.text(0.77, 3.45, "GPT Astra", fontsize=9, color=INK, va="center")
            ax.plot(1.03, 3.45, "o", ms=8, color=ORANGE, clip_on=False); ax.text(1.06, 3.45, "Fable 5.2", fontsize=9, color=INK, va="center")
    fig.savefig(OUT / "5_tools_both.png")
    plt.close(fig)


# List prices, USD per million tokens (input, output), used only to put runs on a common cost axis.
# OpenAI: developers.openai.com/api/docs/pricing, standard tier, prompts under 272k tokens. Haiku: its rate above 100k tokens.
PRICES = {"astra-codex-notools": (10.0, 50.0), "gpt-6.1-sol-codex-notools": (2.0, 10.0), "gpt-6.1-terra-codex-notools": (1.0, 5.0),
          "gpt-6-sol-codex-notools": (2.0, 10.0), "gpt-6-luna-codex-notools": (0.10, 0.50), "gpt-5.6-sol-codex-notools": (4.0, 20.0),
          "gpt-5.6-terra-codex-notools": (2.0, 12.0), "gpt-5.6-luna-codex-notools": (0.20, 1.20),
          "fable-cc-notools": (8.0, 40.0), "opus-cc-notools": (4.0, 20.0), "sonnet-cc-notools": (2.0, 10.0), "haiku-cc-notools": (0.50, 2.50),
          "deepseek-flash-api-notools": (0.30, 1.20), "deepseek-v4-pro-api-notools": (1.32, 3.96)}       # DeepSeek: peak-hour rates
EFFORTS = ["none", "low", "medium", "high", "xhigh", "max"]
BOARD = [("GPT Astra", "astra-codex-notools", BLUE), ("gpt-6.1-sol", "gpt-6.1-sol-codex-notools", BLUE), ("gpt-6.1-terra", "gpt-6.1-terra-codex-notools", BLUE),
         ("gpt-6-sol", "gpt-6-sol-codex-notools", BLUE), ("gpt-6-luna", "gpt-6-luna-codex-notools", BLUE), ("gpt-5.6-sol", "gpt-5.6-sol-codex-notools", BLUE),
         ("gpt-5.6-terra", "gpt-5.6-terra-codex-notools", BLUE), ("gpt-5.6-luna", "gpt-5.6-luna-codex-notools", BLUE),
         ("Claude Fable 5.2", "fable-cc-notools", ORANGE), ("Claude Opus 5.5", "opus-cc-notools", ORANGE),
         ("Claude Sonnet 5.5", "sonnet-cc-notools", ORANGE), ("Claude Haiku 5.5", "haiku-cc-notools", ORANGE),
         ("DeepSeek V4.1 Flash", "deepseek-flash-api-notools", AQUA), ("DeepSeek V4 Pro", "deepseek-v4-pro-api-notools", AQUA)]


def board_points(k: int = 192) -> dict:
    """model key -> [(effort, cost in USD, accuracy %, significant, output tokens)] for the single-request no-tools runs."""
    out = {}
    for _, key, _ in BOARD:
        pts = []
        for e in EFFORTS:
            f = ROOT / f"runs/results/sentences__t15_small__text__{key}__k{k}__{e}.json"
            if not f.exists():
                continue
            c = json.loads(f.read_text())
            u, sc = c.get("usage") or {}, c.get("score")
            if not sc or not u.get("output"):
                continue
            pin, pout = PRICES[key]
            prompt = min(u["prompt"], 240_000 if k else 60_000)   # one pass over the prompt
            pts.append((e, (prompt * pin + u["output"] * pout) / 1e6, 100 * max(0.0, 1 - sc["wer"]), sc["wer_p_len"] <= 0.01, u["output"]))
        out[key] = pts
    return out


def leaderboard() -> None:
    """Accuracy against cost, one line per model through its reasoning levels (no tools, one request)."""
    pts, zero = board_points(192), board_points(0)
    fig = plt.figure(figsize=(8, 5.8), dpi=200)
    fig.text(0.06, 0.95, "No tools: accuracy against cost, by reasoning level", fontsize=16, fontweight="bold", color=INK, va="center")
    fig.text(0.06, 0.902, "One request with 192 labelled examples and 48 held-out trials. 50-word vocabulary.", fontsize=10.5, color=INK2, va="center")
    fig.text(0.06, 0.03, "Cost = tokens at list price. Lines run from low to max effort; larger dot = more effort; hollow = not above chance. Data: Card et al. 2024.",
             fontsize=8, color=MUTED, va="center")
    ax = fig.add_axes([0.10, 0.135, 0.86, 0.715])
    strong, weak = [], []
    for name, key, col in BOARD:
        p = pts.get(key) or []
        if not p:
            continue
        works = any(q[3] for q in p)
        ax.plot([q[1] for q in p], [q[2] for q in p], color=col, lw=1.4 if works else 0.9, ls=(0, (4, 3)), alpha=0.85 if works else 0.4, zorder=2)
        for e, x, y, sig, _ in p:
            ax.plot(x, y, "o", ms=3.5 + 1.2 * EFFORTS.index(e), mfc=col if sig else SURFACE, mec=col, mew=1.3, alpha=1 if works else 0.55, zorder=3)
        for e, x, y, sig, _ in zero.get(key) or []:              # zero-shot: no labelled examples
            ax.plot(x, y, "D", ms=3.6, mfc=SURFACE, mec=col, mew=1.0, alpha=0.6, zorder=2)
        r = max(p, key=lambda q: q[1])                             # label at the right-hand end of the line
        (strong if works else weak).append([r[1], r[2], name])
    for group, gap in ((strong, 4.6), (weak, 3.6)):               # nudge labels apart vertically within each group
        group.sort(key=lambda l: l[1])
        for a in range(1, len(group)):
            for b_ in range(a):
                if abs(np.log10(group[a][0]) - np.log10(group[b_][0])) < 1.0 and group[a][1] - group[b_][1] < gap:
                    group[a][1] = group[b_][1] + gap
    for x, y, text in strong:
        ax.text(x * 1.18, y, text, fontsize=9.5, color=INK, va="center", ha="left", zorder=5)
    import textwrap
    ax.text(0.0042, 62, "Faint lines, at chance at every level:\n" + textwrap.fill(", ".join(t for _, _, t in sorted(weak, key=lambda l: l[2])), 38),
            fontsize=8.5, color=MUTED, va="top", ha="left", linespacing=1.4)
    ax.annotate("zero-shot (no examples):\nevery model at chance", xy=(0.012, 1.5), xytext=(0.0042, 24), fontsize=8.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xscale("log"); ax.set_xlim(0.003, 60); ax.set_ylim(-4, 100)
    ax.set_xticks([0.01, 0.1, 1, 10]); ax.set_xticklabels(["$0.01", "$0.10", "$1", "$10"]); ax.minorticks_off()
    ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.tick_params(axis="both", labelsize=9, colors=MUTED, length=0)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("cost of the request (USD, log scale)", fontsize=10, color=INK2, labelpad=4)
    ax.set_ylabel("words decoded correctly (1 − word error rate)", fontsize=10, color=INK2)
    for i_, (lab, col) in enumerate((("OpenAI", BLUE), ("Anthropic", ORANGE), ("DeepSeek", AQUA))):
        ax.plot(0.0042, 93 - 7 * i_, "o", ms=7, color=col); ax.text(0.0052, 93 - 7 * i_, lab, fontsize=9, color=INK2, va="center")
    fig.savefig(OUT / "7_leaderboard.png")
    plt.close(fig)


def data_norm(s: str) -> str:
    from .data import normalise
    return normalise(s)


def errors_chart() -> None:
    """What predicts which wrong word is decoded (results/errors.json, from icbench.errors)."""
    f = ROOT / "results/errors.json"
    if not f.exists():
        return
    g = json.loads(f.read_text())["groups"]
    rows = [("GPT Astra, in the prompt", "GPT Astra, in the prompt", True), ("Other models above chance, in the prompt", "Other models above chance", True),
            ("Decoders the models trained (tools)", "Decoders trained with tools", True), ("Models at chance, in the prompt", "Models at chance", False),
            ("Activity withheld (guessing)", "Activity withheld (guessing)", False)]
    panels = [("sound", "Sounds alike", (-0.3, 0.6)), ("meaning", "Similar meaning", (-0.3, 0.6)), ("context", "Fits the next word", (0, 6))]
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    frame(fig, "Wrong words fit the sentence and sound like the right word",
          "What predicts which wrong word comes out. 50-word session; bars are 95% intervals.",
          "Choice among the 49 other words; log-odds per standard deviation. Sound: phoneme features. Meaning: GloVe. Data: Card et al. 2024.")
    for n, (key, title, lim) in enumerate(panels):
        ax = fig.add_axes([0.30 + 0.235 * n, 0.21, 0.20, 0.52])
        ax.axvline(0, color=MUTED, lw=1)
        for y, (name, label, neural) in enumerate(rows):
            c = g[name]["choice"][key]
            col = BLUE if neural else MUTED
            ax.plot([c["lo"], c["hi"]], [y, y], color=col, lw=2, alpha=0.55, solid_capstyle="round")
            ax.plot(c["beta"], y, "o", ms=7, mfc=col if neural else SURFACE, mec=col, mew=1.6)
            if n == 0:
                fig.text(0.275, ax.get_position().y1 - (y + 0.5) / len(rows) * ax.get_position().height, label, fontsize=9, color=INK if neural else INK2, ha="right", va="center")
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xlim(*lim)
        ax.set_yticks([])
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=INK2, loc="left", pad=8)
        ax.tick_params(axis="x", labelsize=8, colors=MUTED, length=0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.savefig(OUT / "8_errors.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    learning_curve(); ladder(); examples(); models(); tools_both(); models_tools(); leaderboard(); errors_chart()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
