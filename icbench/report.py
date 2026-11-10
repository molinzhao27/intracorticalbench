"""Collect every finished cell into one table (runs/report.md and runs/report.json).

  uv run python -m icbench.report
"""
from __future__ import annotations

import json

import numpy as np

from .data import normalise
from .models import ROOT, spent
from .score import edits

RESULTS = ROOT / "runs/results"
_ctx_cache: dict = {}


def copied(c: dict) -> float:
    """Share of predictions that are word-for-word one of the labelled sentences in the prompt."""
    from types import SimpleNamespace
    from .run import prepare
    key = (c["dataset"], c["k"])
    if key not in _ctx_cache:
        ds, _, split = c["dataset"].partition("#split")
        a = SimpleNamespace(seed=int(split or 0), nq=48, kmax=192, pcs=12, nq_per_word=6)
        _ctx_cache[key] = {normalise(x[1]) for x in prepare(a, "sentences", ds, c["k"])[1]}
    return float(np.mean([normalise(p) in _ctx_cache[key] for p in c["preds"] if p]))


def paired(a: dict, b: dict, n_boot: int = 10000) -> tuple[float, float, float, float]:
    """WER(a) - WER(b) on the same queries, bootstrap over trials. Returns diff, lo, hi, p(diff >= 0)."""
    m = min(len(a["refs"]), len(b["refs"]))                       # a cell may cover only the first batches
    assert a["refs"][:m] == b["refs"][:m]
    ok = [i for i in range(m) if a["preds"][i] and b["preds"][i]]    # answered in both
    refs = [normalise(a["refs"][i]).split() for i in ok]
    ea = np.array([edits(normalise(a["preds"][i]).split(), r) for i, r in zip(ok, refs)])
    eb = np.array([edits(normalise(b["preds"][i]).split(), r) for i, r in zip(ok, refs)])
    n = np.array([len(r) for r in refs])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(n), (n_boot, len(n)))
    d = (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1)
    return float((ea.sum() - eb.sum()) / n.sum()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float((d >= 0).mean())


def main() -> None:
    cells = {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))}
    rows, lines, unanswered = [], [], []
    lines.append("## Sentences (WER; lower is better)\n")
    lines.append("| data | model | input | labelled trials | effort | WER | chance WER (duration-matched) | p vs chance | WER with activity withheld | activity minus withheld [95% CI] | CER | chance CER | exact | answered | copied from a labelled trial | output tokens / call | cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    key = lambda c: (c["dataset"], c["model"], c["fmt"], c["k"], c["effort"])
    for c in sorted((c for c in cells.values() if c["task"] == "sentences" and c["fmt"] != "none"), key=key):
        s = c["score"]
        if s is None:
            unanswered.append(dict(dataset=c["dataset"], model=c["model"], fmt=c["fmt"], k=c["k"], effort=c["effort"], stops=c["stops"], cost=c["cost"]))
            continue
        ctl = cells.get(c["name"].replace(f"__{c['fmt']}__", "__none__"))
        row = dict(dataset=c["dataset"], model=c["model"], fmt=c["fmt"], k=c["k"], effort=c["effort"], wer=s["wer"],
                   chance=s["wer_chance_len"], p=s["wer_p_len"], cer=s["cer"], cer_chance=s["cer_chance_len"], cer_p=s["cer_p_len"],
                   exact=s["exact"], answered=s["n"], copied=copied(c) if c["k"] else 0.0, out_tokens=float(np.mean(c["out_tokens"])), cost=c["cost"], n=c["n"])
        ctl_txt = diff_txt = "-"
        if ctl and ctl["score"]:
            d, lo, hi, p = paired(c, ctl)
            row.update(withheld_wer=ctl["score"]["wer"], diff=d, diff_lo=lo, diff_hi=hi, diff_p=p)
            ctl_txt, diff_txt = f"{ctl['score']['wer']:.3f}", f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        rows.append(row)
        lines.append(f"| {c['dataset']} | {c['model']} | {c['fmt']} | {c['k']} | {c['effort']} | {s['wer']:.3f} | {s['wer_chance_len']:.3f} | {s['wer_p_len']:.4f} | "
                     f"{ctl_txt} | {diff_txt} | {s['cer']:.3f} | {s['cer_chance_len']:.3f} | {s['exact']:.2f} | {s['n']}/{c['n']} | {row['copied']:.2f} | {row['out_tokens']:.0f} | ${c['cost']:.2f} |")
    for u in unanswered:
        lines.append(f"\nNo answers: {u['dataset']} {u['model']} {u['fmt']} k={u['k']} {u['effort']} (stop reasons {sorted(set(u['stops']))}, ${u['cost']:.2f})")
    lines.append("\n## Seven-word classification (accuracy; chance 0.143)\n")
    lines.append("| participant | model | input | labelled trials | effort | accuracy | p vs chance | output tokens / call | cost |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    words = []
    for c in sorted((c for c in cells.values() if c["task"] == "words"), key=key):
        s = c["score"]
        words.append(dict(dataset=c["dataset"], model=c["model"], fmt=c["fmt"], k=c["k"], effort=c["effort"], acc=s["acc"], p=s["p"], n=c["n"],
                          out_tokens=float(np.mean(c["out_tokens"])), cost=c["cost"]))
        lines.append(f"| {c['dataset']} | {c['model']} | {c['fmt']} | {c['k']} | {c['effort']} | {s['acc']:.3f} ({round(s['acc'] * c['n'])}/{c['n']}) | {s['p']:.2g} | {np.mean(c['out_tokens']):.0f} | ${c['cost']:.2f} |")
    base = ROOT / "runs/baselines.json"
    if base.exists():
        b = json.loads(base.read_text())
        lines.append("\n## Classical decoders on the same input\n")
        lines.append("| data | labelled trials | decoder | WER | chance WER | p | CER | chance CER |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in b["sentences"]:
            s = r["score"]
            lines.append(f"| {r['dataset']} | {r['k']} | {r['decoder']} | {s['wer']:.3f} | {s['wer_chance_len']:.3f} | {s['wer_p_len']:.4f} | {s['cer']:.3f} | {s['cer_chance_len']:.3f} |")
        lines.append("\n| participant | nearest centroid | LDA | labelled trials |\n|---|---|---|---|")
        for w in b["words"]:
            lines.append(f"| {w['participant']} | {w['nearest_centroid']:.3f} | {w['lda']:.3f} | {w['n_context']} |")
    lines.append(f"\nSpent: {json.dumps({k: round(v, 2) for k, v in spent().items() if isinstance(v, float)})}")
    (ROOT / "runs/report.md").write_text("\n".join(lines) + "\n")
    (ROOT / "runs/report.json").write_text(json.dumps(dict(sentences=rows, words=words, unanswered=unanswered), indent=1))
    out = ROOT / "results"                                        # tracked copy: runs/ stays local
    (out / "cells").mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(l for l in lines if not l.startswith("\nSpent:")) + "\n")
    (out / "report.json").write_text(json.dumps(dict(sentences=rows, words=words, unanswered=unanswered), indent=1))
    keep = ("name", "task", "dataset", "fmt", "model", "k", "effort", "n", "preds", "refs", "missing", "stops", "score")
    for name, c in cells.items():
        (out / "cells" / f"{name}.json").write_text(json.dumps({k: c[k] for k in keep if k in c}, indent=1))
    if base.exists():
        (out / "baselines.json").write_text(base.read_text())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
