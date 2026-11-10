"""Run one benchmark cell: task x dataset x input format x model x number of labelled examples x effort.

  uv run python -m icbench.run sentences --dataset t15_small --model fable --k 0 16 64 192 --effort low high
  uv run python -m icbench.run words --dataset t15 --model fable --effort low high
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import data, models
from .score import score

RESULTS = models.ROOT / "runs/results"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"                          # first ten are the default deciles

SETUP = """You are the decoder in a research benchmark on speech brain-computer interfaces.

SETUP
A research participant with severe dysarthria has microelectrode arrays implanted in speech motor cortex (precentral gyrus). On each trial they are shown {what} and {behaviour}. Threshold-crossing spike counts and spike-band power are recorded on every electrode.

DATA FORMAT
{fmt}

TASK
For each query trial, output {target}. Always commit to a best guess for every query: a guess always scores at least as well as no answer. End your reply with exactly one line per query, in the form `Q1: <answer>`."""

FMT = {
    "text": "Each trial is a sequence of time frames, {ms} ms per frame, running from the go cue to the end of the trial, one frame per line, earliest first. A frame is {k} letters, one per principal component of the population activity (PCA over all electrodes' z-scored features; the same {k} components in the same order for every trial here). A letter is that component's decile over the whole session: A = lowest 10%, J = highest 10%.",
    "image": "Each trial is an image. Time runs left to right ({ms} ms per column) from the go cue to the end of the trial. Each row is one principal component of the population activity (PCA over all electrodes' z-scored features; the same {k} components in the same order, top to bottom, for every trial here). Colour is that component's decile over the whole session on the viridis scale: dark purple = lowest 10%, yellow = highest 10%.",
    "none": "The neural activity is withheld in this control condition. You are given only each trial's duration in seconds.",
    "vector": "Each trial is one line of {k} letters: the activity averaged over the 1.5 s after the go cue, one letter per principal component of the population activity (PCA over all electrodes' z-scored features; the same {k} components in the same order for every trial here). A letter is that component's decile over all trials: A = lowest 10%, J = highest 10%.",
}


def describe(fmt: str, dt: float, pcs: int, levels: int = 10) -> str:
    """The DATA FORMAT paragraph. With the default ten levels the wording is unchanged."""
    text = FMT[fmt].format(ms=int(dt * 1000), k=pcs)
    if levels != 10 and fmt == "text":
        old = "A letter is that component's decile over the whole session: A = lowest 10%, J = highest 10%."
        assert old in text
        text = text.replace(old, f"A letter is that component's level over the whole session, in {levels} equally populated steps: "
                                 f"A = lowest, {LETTERS[levels - 1]} = highest.")
    return text


def encoding_tag(a) -> str:
    """Suffix for result names when the encoding differs from the default (12 components, deciles, dataset frame)."""
    parts = [f"pc{a.pcs}"] * (a.pcs != 12) + [f"lv{a.levels}"] * (getattr(a, "levels", 10) != 10) + [f"fr{a.frame}"] * bool(getattr(a, "frame", None))
    return "-".join(parts)


def render(q: np.ndarray, fmt: str) -> list[dict]:
    if fmt == "image":
        return [{"type": "image", "b64": data.as_png(q)}]
    if fmt == "none":
        return []
    return [{"type": "text", "text": "\n".join(" ".join(LETTERS[v] for v in row) for row in q) + "\n"}]


def merge(blocks: list[dict]) -> list[dict]:
    out: list[dict] = []
    for b in blocks:
        if b["type"] == "text" and out and out[-1]["type"] == "text":
            out[-1] = {"type": "text", "text": out[-1]["text"] + b["text"]}
        else:
            out.append(b)
    return out


def build(ctx: list[tuple], queries: list[tuple], fmt: str, noun: str, cross: bool = False) -> list[dict]:
    """ctx / queries: (quantised activity, label, seconds). Context is the cacheable prefix."""
    blocks: list[dict] = []
    if ctx:
        where = "AN EARLIER SESSION (A DIFFERENT DAY)" if cross else "THE SAME SESSION"
        blocks.append({"type": "text", "text": f"LABELLED TRIALS FROM {where} ({len(ctx)})\n\n"})
        for i, (q, label, sec) in enumerate(ctx, 1):
            blocks.append({"type": "text", "text": f"Trial {i} ({sec:.1f} s)\n{noun}: {label}\n"})
            blocks += render(q, fmt)
            blocks.append({"type": "text", "text": "\n"})
        blocks = merge(blocks) + [{"type": "cache"}]
    when = "from a later session than the labelled trials" if cross else "same session"
    blocks.append({"type": "text", "text": f"QUERY TRIALS ({len(queries)}), {when}, {noun.lower()} unknown\n\n"})
    for i, (q, _, sec) in enumerate(queries, 1):
        blocks.append({"type": "text", "text": f"Q{i} ({sec:.1f} s)\n"})
        blocks += render(q, fmt)
        blocks.append({"type": "text", "text": "\n"})
    return merge(blocks)


def parse(text: str, n: int) -> list[str]:
    found = {int(a): b.strip() for a, b in re.findall(r"^\**Q(\d+)\**\s*:\s*(.*)$", text or "", flags=re.M)}
    return [found.get(i, "") for i in range(1, n + 1)]


def prepare(a, task: str, dataset: str, k: int):
    """Fixed split and features for a dataset. Returns (queries, context, frame seconds, wording)."""
    rng = np.random.default_rng(a.seed)
    levels, frame = getattr(a, "levels", 10), getattr(a, "frame", None)   # encoding options; defaults = the main benchmark
    if task == "sentences":
        base = dataset.split(":")[-1]
        if ">" in base:                                           # "A>B": labelled trials from session A, queries from B
            src, dst = base.split(">")
            lab, dt = data.load_sentences(src, frame)
            tst, _ = data.load_sentences(dst, frame)
            used = [tst[i] for i in rng.permutation(len(tst))[:a.nq]] + [lab[i] for i in rng.permutation(len(lab))[:a.kmax]]
        else:
            trials, dt = data.load_sentences(base, frame)
            order = rng.permutation(len(trials))
            qi, pool = order[:a.nq], order[a.nq:]
            used = [trials[i] for i in np.r_[qi, pool[:a.kmax]]]  # features fixed across k
        qi = range(a.nq)
        if dataset.startswith("planted_"):
            q = data.plant(used, a.pcs, dataset.split(":")[0].split("_")[1])
        else:
            q = data.quantise(used, a.pcs, levels)
        beh = data.BEHAVIOUR.get(base.split(">")[-1])
        wording = ("an English sentence", "the sentence the participant was " + ("imagining" if beh else "attempting to say"), "Sentence")
    else:
        trials, dt = data.load_words(dataset), 1.5
        order = rng.permutation(len(trials))
        labels = np.array([t.label for t in trials])
        qi = np.concatenate([[i for i in order if labels[i] == w][:a.nq_per_word] for w in sorted(set(labels))])
        qi = rng.permutation(qi)
        pool = np.array([i for i in order if i not in set(qi)])
        used = [trials[i] for i in np.r_[qi, pool]]
        q = data.quantise(used, a.pcs)
        k = len(pool) if k else 0
        wording = ("one of seven words (ban, choice, day, feel, kite, though, were)",
                   "which of the seven words the participant was attempting to say", "Word")
    nq = len(qi)
    items = [(q[i], used[i].label, used[i].seconds) for i in range(len(used))]
    return items[:nq], items[nq:nq + k], items[nq:], dt, wording


def run_cell(a, task: str, dataset: str, model: str, k: int, effort: str) -> dict:
    queries, ctx, pool, dt, (what, target, noun) = prepare(a, task, dataset, k)
    nq, k = len(queries), len(ctx)
    if a.seed:                                                    # a different random split is its own dataset row
        dataset = f"{dataset}#split{a.seed}"
    fmt_name = a.fmt + ("-" + encoding_tag(a) if encoding_tag(a) else "")
    name = f"{task}__{dataset.replace('>', '-to-')}__{fmt_name}__{model}__k{k if task == 'sentences' else min(k, 1)}__{effort}"
    path = RESULTS / f"{name}.json"
    fmt = "vector" if (task == "words" and a.fmt == "text") else a.fmt
    base = dataset.split(":")[-1].split("#")[0]
    cross = ">" in base
    behaviour = data.BEHAVIOUR.get(base.split(">")[-1], "attempt to say it aloud")
    system = SETUP.format(what=what, target=target, behaviour=behaviour, fmt=describe(fmt, dt, a.pcs, getattr(a, "levels", 10)))
    if task == "sentences":                                       # session facts from non-query trials
        nw = [len(x[1].split()) for x in pool]
        sec = [x[2] for x in pool]
        system += (f"\n\nSESSION FACTS\nThe participant speaks slowly. Sentences in this session are {min(nw)}-{max(nw)} words "
                   f"long (median {int(np.median(nw))}); trials last {min(sec):.0f}-{max(sec):.0f} s (median {np.median(sec):.0f} s).")
    batches = [queries[i:i + a.batch] for i in range(0, nq, a.batch)][:a.max_batches]
    queries = [x for b in batches for x in b]
    nq = len(queries)
    prompts = [build(ctx, b, a.fmt, noun, cross) for b in batches]
    if a.dry:
        chars = sum(len(b.get("text", "")) for b in prompts[0])
        print(f"{name}: {len(prompts)} calls, ~{chars // 2:,} text tokens each" + (f", {sum(b['type'] == 'image' for b in prompts[0])} images" if a.fmt == "image" else ""))
        return {}
    first = models.call(model, effort, system, prompts[0], tag=name)   # warms the provider cache
    with ThreadPoolExecutor(a.workers) as ex:
        rest = list(ex.map(lambda p: models.call(model, effort, system, p, tag=name), prompts[1:]))
    outs = [first] + rest
    preds = [p for o, b in zip(outs, batches) for p in parse(o["text"], len(b))]
    refs = [x[1] for x in queries]
    res = dict(name=name, task=task, dataset=dataset, fmt=fmt_name, model=model, k=k, effort=effort, n=nq,
               preds=preds, refs=refs, missing=sum(not p for p in preds),
               stops=[o["stop"] for o in outs], refusals=[o.get("refusal") for o in outs if o["stop"] == "refusal"], cost=sum(o["cost"] for o in outs),
               out_tokens=[o["usage"]["output"] for o in outs],
               in_tokens=[o["usage"]["input"] + o["usage"]["cache_read"] + o["usage"]["cache_write"] for o in outs])
    if task == "sentences":
        ok = [i for i, p in enumerate(preds) if p]                 # unanswered trials are reported, not scored
        res["score"] = s = score([preds[i] for i in ok], [refs[i] for i in ok], [queries[i][2] for i in ok]) if len(ok) > 1 else None
        line = "no answers to score" if s is None else (f"WER {s['wer']:.3f} (chance {s['wer_chance']:.3f}, p={s['wer_p']:.4f}; duration-matched "
                f"{s['wer_chance_len']:.3f}, p={s['wer_p_len']:.4f})  CER {s['cer']:.3f} (chance {s['cer_chance']:.3f}, p={s['cer_p']:.4f})")
    else:
        from scipy.stats import binomtest
        correct = sum(data.normalise(p) == r for p, r in zip(preds, refs))
        res["score"] = dict(acc=correct / nq, chance=1 / 7, p=float(binomtest(correct, nq, 1 / 7, alternative="greater").pvalue))
        line = f"accuracy {correct}/{nq} = {correct / nq:.3f} (chance 0.143, p={res['score']['p']:.4g})"
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    print(f"{name}: {line}  | missing {res['missing']}  out tok/call {int(np.mean(res['out_tokens']))}  ${res['cost']:.2f}", flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["sentences", "words"])
    p.add_argument("--dataset", nargs="+", required=True)
    p.add_argument("--model", nargs="+", default=["fable"])
    p.add_argument("--k", nargs="+", type=int, default=[0])
    p.add_argument("--effort", nargs="+", default=["high"])
    p.add_argument("--fmt", default="text", choices=["text", "image", "none"])
    p.add_argument("--pcs", type=int, default=12)
    p.add_argument("--levels", type=int, default=10, help="quantile levels per component (10 = deciles, up to 26)")
    p.add_argument("--frame", type=int, default=None, help="source steps averaged per frame (default: dataset setting)")
    p.add_argument("--nq", type=int, default=48)
    p.add_argument("--nq-per-word", type=int, default=6)
    p.add_argument("--kmax", type=int, default=192)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--max-batches", type=int, default=None, help="only the first N batches of queries (cheaper cells)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    for ds in a.dataset:
        for model in a.model:
            for k in a.k:
                for effort in a.effort:
                    try:
                        run_cell(a, a.task, ds, model, k, effort)
                    except models.BudgetExceeded as e:
                        print("STOPPED:", e)
                        return
    print("spent so far:", json.dumps(models.spent()))


if __name__ == "__main__":
    main()
