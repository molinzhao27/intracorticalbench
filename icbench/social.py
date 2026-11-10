"""Static PNG figures (1600 x 900) for posts and the README.

  uv run python -m icbench.social        # writes assets/figures/*.png
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import ROOT
from .run import LETTERS, prepare

OUT = ROOT / "assets/figures"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # validated categorical slots 1-3
RAMP = ["#86b6ef", "#2a78d6", "#104281"]                        # one hue, light to dark, for ordered conditions
GOOD, BAD = "#006300", "#d03b3b"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


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
          "GPT Astra, high effort. 192 labelled trials in the prompt, then held-out trials. Plain model call: no tools, no training.",
          "First 14 of 48 held-out sentences (participant T15, 50-word vocabulary; data: Card et al. 2024). Each letter is one principal component's decile in a 160 ms frame; "
          "twelve components per frame.")
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


def data_norm(s: str) -> str:
    from .data import normalise
    return normalise(s)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    learning_curve(); ladder(); examples()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
