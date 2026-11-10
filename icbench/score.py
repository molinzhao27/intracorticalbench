"""WER / CER and the permutation nulls that define "chance" for a given set of predictions."""
from __future__ import annotations

import numpy as np

from .data import normalise


def edits(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(preds: list[str], refs: list[str], seconds: list[float], n_perm: int = 10000, seed: int = 0) -> dict:
    """Corpus WER/CER of preds vs refs, plus two nulls built by re-pairing the same predictions
    with other trials' references:

    - shuffled: any other trial. This is what a model gets from the language prior alone.
    - duration-matched: only among the 4 trials closest in duration. Trial duration is the one
      thing a model can read without decoding anything, so beating this null means the model used
      more than how long the trial lasted.
    p = share of permutations whose error rate is <= the observed one.
    """
    p = [normalise(x) for x in preds]
    r = [normalise(x) for x in refs]
    n = len(r)
    out = {"n": n}
    rng = np.random.default_rng(seed)
    by_dur = np.argsort(seconds)
    groups = [by_dur[i:i + 4] for i in range(0, n, 4)]
    for name, tok in (("wer", str.split), ("cer", list)):
        pt, rt = [tok(x) for x in p], [tok(x) for x in r]
        e = np.array([[edits(a, b) for b in rt] for a in pt])    # e[i, j]: pred i vs ref j
        denom = sum(len(x) for x in rt)
        obs = np.trace(e) / denom
        idx = np.arange(n)
        full = np.array([e[idx, rng.permutation(n)].sum() for _ in range(n_perm)]) / denom
        matched = np.empty(n_perm)
        for k in range(n_perm):
            perm = idx.copy()
            for g in groups:
                perm[g] = rng.permutation(g)
            matched[k] = e[idx, perm].sum() / denom
        out[name] = float(obs)
        out[f"{name}_chance"] = float(full.mean())
        out[f"{name}_p"] = float(((full <= obs).sum() + 1) / (n_perm + 1))
        out[f"{name}_chance_len"] = float(matched.mean())
        out[f"{name}_p_len"] = float(((matched <= obs).sum() + 1) / (n_perm + 1))
    out["exact"] = float(np.mean([a == b for a, b in zip(p, r)]))
    out["wordcount_mae"] = float(np.mean([abs(len(a.split()) - len(b.split())) for a, b in zip(p, r)]))
    return out
