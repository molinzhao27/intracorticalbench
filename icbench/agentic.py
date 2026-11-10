"""The same decoding task handed to a coding agent with tools (e.g. Codex CLI) instead of a bare
model call: labelled trials and unlabelled test trials as files, the agent may write and run code.

  uv run python -m icbench.agentic make  t15_small /path/to/taskdir --python /path/to/venv/bin/python
  codex exec ... -C /path/to/taskdir "$(cat /path/to/taskdir/TASK.md)" --json > /path/to/taskdir.log
  uv run python -m icbench.agentic score t15_small /path/to/taskdir --log /path/to/taskdir.log --model astra-codex

The test file holds no sentences and its trials are shuffled under random ids; the answer key stays
in this repo (runs/agentic_keys). `score` also audits the agent's command log for anything that
touched files outside the task folder, since the original labelled data is on the same disk.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .models import ROOT
from .run import RESULTS, prepare
from .score import score

KEYS = ROOT / "runs/agentic_keys"

TASK = """# Decode attempted speech from intracortical recordings

A research participant with severe dysarthria has microelectrode arrays implanted in speech motor
cortex (precentral gyrus). On each trial they were shown an English sentence and {behaviour}.
Threshold-crossing spike counts and spike-band power were recorded on every electrode.

## Files in this folder

- `train.jsonl`: {n_train} labelled trials from one session. Fields: `id`, `seconds`, `sentence`, `activity`.
- `test.jsonl`: {n_test} trials from {test_from} with the sentence removed. Fields: `id`, `seconds`, `activity`.
{extra_files}
`activity` is a list of time frames, {ms} ms per frame, from the go cue to the end of the trial.
Each frame is {k} integers from 0 to 9: one per principal component of the population activity (PCA
over all electrodes' z-scored features; the same {k} components in the same order for every trial),
each given as that component's decile over all trials here (0 = lowest 10%, 9 = highest 10%).

Session facts: the participant speaks slowly. Sentences in this session are {wmin}-{wmax} words long
(median {wmed}); trials last {smin:.0f}-{smax:.0f} s (median {smed:.0f} s).

## Goal

Write `predictions.jsonl` with one line per test trial: `{{"id": "...", "sentence": "..."}}`.
Predictions are scored by word error rate against the true sentences (case and punctuation ignored).
Give a best-guess sentence for every test trial.

## Rules

- Use only the files in this folder. Do not read or search anything outside it, and do not use the
  network. The run is logged and audited for this.
{rules}
- When finished, also write `NOTES.md` (15 lines at most): what you did and how confident you are.
"""

FULL = """- `full_train.npz`, `full_test.npz`: the full recording for the same trials, far more detailed than
  `activity`. One array per trial id (`numpy.load(file)[trial_id]`), shape (time steps, {nfeat}), {step} ms
  per step from the go cue to the end of the trial, z-scored per feature. Columns 0-{h1} are
  threshold-crossing spike counts and columns {h}-{l} are spike-band power for the same {h} electrodes
  in the same order.{arrays}
"""

RULES = {
    # the agent may build any decoder it likes
    "full": """- Inside those limits, do whatever gives the lowest word error rate: write and run code, train
  models, inspect the data yourself, or combine these. Python with numpy, scipy, scikit-learn and
  torch is at `{python}`. Keep total compute to roughly 20 minutes on this laptop (CPU).""",
    # code as a viewer and calculator only: the model itself has to do the decoding
    # as "lite", but any label-free manipulation of the data is welcome; still nothing learned or fitted
    "manipulate": """- You are the decoder. Work out each sentence yourself, by studying the labelled trials and
  comparing them with the test trials.
- You may write and run code to read and manipulate the data in any way that only transforms or
  summarises it: slicing, averaging over time or electrodes, smoothing, normalising, subtracting
  baselines, combining or projecting features, correlating, computing distances, sorting numbers,
  printing. Use as much or as little of the detailed recording as you find useful. Python with
  numpy only is at `{python}`.
- No machine learning. Nothing may be fitted or trained to predict words: no classifiers,
  regression, neural networks, hidden Markov models, EM, clustering, or nearest-neighbour and
  template classifiers. No program may assign words or sentences to test trials. Where a trial is
  cut into words, which labelled examples to compare with, and what each word is are your
  decisions; code only supplies the numbers you ask for. Every command should finish within a
  few seconds.
- Type the sentences in `predictions.jsonl` out yourself; they must be your own decisions, not the
  output of a program. In `NOTES.md`, also say exactly what you used code for.""",
    "lite": """- You are the decoder. Work out each sentence yourself, by studying the labelled trials and
  comparing them with the test trials.
- You may run code, but only as a viewer and calculator: load and print trials, reformat them, cut
  out the frames you want to look at, and compute simple descriptive numbers (means, differences,
  correlations or distances between segments you chose). Python with numpy only is at `{python}`.
- Not allowed: fitting or training any model or automated decoder. That means no neural networks,
  hidden Markov models, EM, regression, classifiers, clustering, alignment or search algorithms, or
  optimisation loops, and no script that outputs predicted words or sentences. Every command
  should finish within a few seconds.
- Type the sentences in `predictions.jsonl` out yourself; they must be your own decisions, not the
  output of a program. In `NOTES.md`, also say exactly what you used code for.""",
}


def _args(seed: int = 0, pcs: int = 12, levels: int = 10, frame: int | None = None):
    return SimpleNamespace(seed=seed, nq=48, kmax=192, pcs=pcs, nq_per_word=6, levels=levels, frame=frame)


def make(dataset: str, out: Path, python: str, k: int = 192, rules: str = "full", full: bool = False) -> None:
    queries, ctx, pool, dt, _ = prepare(_args(), "sentences", dataset, k)
    k = len(ctx)                                                   # smaller sessions have fewer labelled trials
    out.mkdir(parents=True, exist_ok=True)
    import zlib
    rng = np.random.default_rng(zlib.crc32(str(out.resolve()).encode()))   # test ids and order differ per task folder
    extra_files = ""
    if full:                                                       # the same trials at the recording's own resolution
        from . import data as D
        raw, step = D.load_sentences(dataset, 1)
        o = np.random.default_rng(0).permutation(len(raw))         # the split prepare() makes
        q_raw, c_raw = [raw[i] for i in o[:48]], [raw[i] for i in o[48:48 + k]]
        assert [t.label for t in q_raw] == [x[1] for x in queries] and [t.label for t in c_raw] == [x[1] for x in ctx]
        nfeat = raw[0].x.shape[1]
        arrays = {"t15": " The electrodes are four 64-channel arrays, in order: ventral area 6v, area 4, area 55b, dorsal area 6v.",
                  "t16": " The electrodes are, in order: 64 in PEF, 64 in ventral area 6v, 128 in area 6d."}.get(dataset[:3], "")
        extra_files = FULL.format(nfeat=nfeat, step=int(step * 1000), h=nfeat // 2, h1=nfeat // 2 - 1, l=nfeat - 1, arrays=arrays)
    with open(out / "train.jsonl", "w") as f:
        for i, (q, label, sec) in enumerate(ctx, 1):
            f.write(json.dumps(dict(id=f"train{i:03d}", seconds=round(sec, 2), sentence=label, activity=q.tolist())) + "\n")
    ids = [f"test{n:04d}" for n in rng.choice(9000, len(queries), replace=False) + 1000]
    order = rng.permutation(len(queries))                          # file order unrelated to query order
    if full:
        np.savez(out / "full_train.npz", **{f"train{i:03d}": t.x.astype(np.float16) for i, t in enumerate(c_raw, 1)})
        np.savez(out / "full_test.npz", **{ids[j]: q_raw[j].x.astype(np.float16) for j in order})
    with open(out / "test.jsonl", "w") as f:
        for j in order:
            q, _, sec = queries[j]
            f.write(json.dumps(dict(id=ids[j], seconds=round(sec, 2), activity=q.tolist())) + "\n")
    KEYS.mkdir(parents=True, exist_ok=True)
    (KEYS / f"{out.name}.json").write_text(json.dumps(dict(dataset=dataset, k=k, ids=ids)))
    nw, sec = [len(x[1].split()) for x in pool], [x[2] for x in pool]
    from . import data
    base = dataset.split(">")[-1]
    behaviour = {"attempt": "attempted to say it aloud"}.get("attempt") if base not in data.BEHAVIOUR else "imagined saying it (inner speech, no movement or sound)"
    test_from = "a later session (a different day, two days after the labelled one)" if ">" in dataset else "the same session"
    (out / "TASK.md").write_text(TASK.format(n_train=len(ctx), n_test=len(queries), ms=int(dt * 1000), k=12, python=python,
                                             behaviour=behaviour, test_from=test_from, rules=RULES[rules].format(python=python), extra_files=extra_files,
                                             wmin=min(nw), wmax=max(nw), wmed=int(np.median(nw)),
                                             smin=min(sec), smax=max(sec), smed=float(np.median(sec))))
    print(f"wrote {out}: {len(ctx)} labelled trials, {len(queries)} test trials")


def audit(log: Path, taskdir: Path, python: str) -> dict:
    """List every shell command the agent ran and flag any that mention a path outside the task folder."""
    cmds = []
    for line in log.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = ev.get("item") or {}
        if ev.get("type") == "item.completed" and item.get("type") == "command_execution":
            cmds.append(item.get("command", ""))
    allowed = [str(taskdir), str(taskdir.resolve()), str(Path(python).parent.parent)]
    flagged = []
    for c in cmds:
        stripped = c
        for a in allowed:
            stripped = stripped.replace(a, "")
        if re.search(r"/Users/|~/|\.\./|/Volumes|/private/|/tmp/|mdfind|locate |find /|curl |wget |pip install|hdf5|\.mat\b", stripped):
            flagged.append(c[:300])
    return dict(n_commands=len(cmds), flagged=flagged)


def do_score(dataset: str, taskdir: Path, model: str, effort: str, log: Path | None, python: str) -> None:
    key = json.loads((KEYS / f"{taskdir.name}.json").read_text())
    queries, _, _, _, _ = prepare(_args(), "sentences", dataset, key["k"])
    pred = {}
    for line in (taskdir / "predictions.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            pred[d["id"]] = d.get("sentence", "")
    preds = [pred.get(i, "") for i in key["ids"]]
    refs, secs = [x[1] for x in queries], [x[2] for x in queries]
    ok = [i for i, p in enumerate(preds) if p]
    s = score([preds[i] for i in ok], [refs[i] for i in ok], [secs[i] for i in ok])
    name = f"sentences__{dataset.replace('>', '-to-')}__text__{model}__k{key['k']}__{effort}"
    res = dict(name=name, task="sentences", dataset=dataset, fmt="text", model=model, k=key["k"], effort=effort, n=len(refs),
               preds=preds, refs=refs, missing=len(refs) - len(ok), stops=["agent"], refusals=[], cost=0.0, out_tokens=[0], in_tokens=[0], score=s)
    if log:
        res["audit"] = audit(log, taskdir, python)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(f"{name}: WER {s['wer']:.3f} (duration-matched chance {s['wer_chance_len']:.3f}, p={s['wer_p_len']:.4f})  CER {s['cer']:.3f}  "
          f"exact {s['exact']:.2f}  answered {len(ok)}/{len(refs)}")
    if log:
        print(f"audit: {res['audit']['n_commands']} commands, {len(res['audit']['flagged'])} flagged")
        for c in res["audit"]["flagged"]:
            print("   FLAG:", c)


def make_images(dataset: str, out: Path, k: int) -> None:
    """Image mode through an agent harness, without tools: every trial is one heat-map image attached
    to the prompt (labelled trials first, then the queries); the sentences are listed in the text."""
    import base64
    from . import data
    from .run import FMT, SETUP
    queries, ctx, pool, dt, (what, target, noun) = prepare(_args(), "sentences", dataset, k)
    (out / "img").mkdir(parents=True, exist_ok=True)
    paths = []
    for i, (q, _, _) in enumerate(ctx + queries, 1):
        f = out / "img" / f"{i:03d}.png"
        f.write_bytes(base64.b64decode(data.as_png(q)))
        paths.append(str(f))
    nw, sec = [len(x[1].split()) for x in pool], [x[2] for x in pool]
    text = SETUP.format(what=what, target=target, behaviour=data.BEHAVIOUR.get(dataset, "attempt to say it aloud"),
                        fmt=FMT["image"].format(ms=int(dt * 1000), k=12))
    text += (f"\n\nSESSION FACTS\nThe participant speaks slowly. Sentences in this session are {min(nw)}-{max(nw)} words "
             f"long (median {int(np.median(nw))}); trials last {min(sec):.0f}-{max(sec):.0f} s (median {np.median(sec):.0f} s).")
    text += (f"\n\nThe {len(paths)} attached images are in this order: images 1-{len(ctx)} are labelled trials from the same "
             f"session, images {len(ctx) + 1}-{len(paths)} are the query trials Q1-Q{len(queries)}.\n\nLABELLED TRIALS\n")
    text += "".join(f"Image {i} ({sec:.1f} s): {label}\n" for i, (_, label, sec) in enumerate(ctx, 1))
    text += "\nQUERY TRIALS\n" + "".join(f"Image {len(ctx) + i} = Q{i} ({sec:.1f} s)\n" for i, (_, _, sec) in enumerate(queries, 1))
    text += ("\nAnswer from the attached images alone. Do not run any shell commands, code or other tools, and do not "
             "read or write files; this run is audited for that. Reply with your answers only.")
    (out / "prompt.txt").write_text(text)
    (out / "images.txt").write_text("\n".join(paths) + "\n")
    print(f"wrote {out}: {len(ctx)} labelled + {len(queries)} query images")


def make_text(dataset: str, out: Path, k: int, a) -> None:
    """The in-prompt text test sent through an agent harness with tools forbidden: one request holding
    the labelled trials and all held-out trials. Used to try richer encodings without API cost."""
    from . import data
    from .run import build, describe
    from .run import SETUP as S
    queries, ctx, pool, dt, (what, target, noun) = prepare(a, "sentences", dataset, k)
    out.mkdir(parents=True, exist_ok=True)
    nw, sec = [len(x[1].split()) for x in pool], [x[2] for x in pool]
    text = S.format(what=what, target=target, behaviour=data.BEHAVIOUR.get(dataset, "attempt to say it aloud"),
                    fmt=describe("text", dt, a.pcs, a.levels))
    text += (f"\n\nSESSION FACTS\nThe participant speaks slowly. Sentences in this session are {min(nw)}-{max(nw)} words "
             f"long (median {int(np.median(nw))}); trials last {min(sec):.0f}-{max(sec):.0f} s (median {np.median(sec):.0f} s).\n\n")
    text += "".join(b["text"] for b in build(ctx, queries, "text", noun) if b["type"] == "text")
    text += ("\nAnswer from the data above alone. Do not run any shell commands, code or other tools, and do not read or "
             "write files; this run is audited for that. Reply with your answers only.")
    (out / "prompt.txt").write_text(text)
    print(f"wrote {out}: {len(ctx)} labelled + {len(queries)} query trials, about {len(text) // 2:,} tokens")


def score_images(dataset: str, out: Path, k: int, model: str, effort: str, log: Path | None, a=None, fmt: str = "image") -> None:
    from .run import parse
    queries, ctx, _, _, _ = prepare(a or _args(), "sentences", dataset, k)
    preds = parse((out / "answer.txt").read_text(), len(queries))
    refs, secs = [x[1] for x in queries], [x[2] for x in queries]
    ok = [i for i, p in enumerate(preds) if p]
    s = score([preds[i] for i in ok], [refs[i] for i in ok], [secs[i] for i in ok])
    name = f"sentences__{dataset}__{fmt}__{model}__k{len(ctx)}__{effort}"
    res = dict(name=name, task="sentences", dataset=dataset, fmt=fmt, model=model, k=len(ctx), effort=effort, n=len(refs),
               preds=preds, refs=refs, missing=len(refs) - len(ok), stops=["agent"], refusals=[], cost=0.0, out_tokens=[0], in_tokens=[0], score=s)
    if log:
        res["audit"] = audit(log, out, "python")
        res["audit"]["tools_used"] = res["audit"]["n_commands"] > 0
    (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=1))
    print(f"{name}: WER {s['wer']:.3f} (duration-matched chance {s['wer_chance_len']:.3f}, p={s['wer_p_len']:.4f})  CER {s['cer']:.3f}  "
          f"exact {s['exact']:.2f}  answered {len(ok)}/{len(refs)}" + (f"  | commands run: {res['audit']['n_commands']}" if log else ""))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["make", "score", "images", "score-images", "text", "score-text"])
    p.add_argument("--pcs", type=int, default=12)
    p.add_argument("--levels", type=int, default=10)
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--k", type=int, default=192)
    p.add_argument("dataset")
    p.add_argument("taskdir", type=Path)
    p.add_argument("--python", default="python")
    p.add_argument("--model", default="astra-codex")
    p.add_argument("--effort", default="high")
    p.add_argument("--log", type=Path)
    p.add_argument("--rules", default="full", choices=list(RULES))
    p.add_argument("--full-recording", action="store_true", help="also give the unreduced recording as .npz files")
    a = p.parse_args()
    enc = _args(pcs=a.pcs, levels=a.levels, frame=a.frame)
    if a.cmd == "text":
        make_text(a.dataset, a.taskdir, a.k, enc)
    elif a.cmd == "score-text":
        from .run import encoding_tag
        score_images(a.dataset, a.taskdir, a.k, a.model, a.effort, a.log, enc, "text" + ("-" + encoding_tag(enc) if encoding_tag(enc) else ""))
    elif a.cmd == "images":
        make_images(a.dataset, a.taskdir, a.k)
    elif a.cmd == "score-images":
        score_images(a.dataset, a.taskdir, a.k, a.model, a.effort, a.log)
    elif a.cmd == "make":
        make(a.dataset, a.taskdir, a.python, rules=a.rules, full=a.full_recording)
    else:
        do_score(a.dataset, a.taskdir, a.model, a.effort, a.log, a.python)


if __name__ == "__main__":
    main()
