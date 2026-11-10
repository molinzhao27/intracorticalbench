"""How much does the text encoding throw away?

Two views, on one session with the benchmark's own split (192 labelled trials, 48 held-out):
1. Signal: share of the recording's variance that survives each step (time averaging, PCA, deciles).
2. Decodable information: the same small decoder (character GRU-CTC, trained from scratch on the
   labelled trials) given the input at each stage, plus two cheap alternatives to the fed format.

  uv run python -m icbench.lossiness [t15_small]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import data
from .baselines import ctc
from .models import ROOT
from .score import score


def raw_trials(name: str):
    """Unbinned trials (20 ms steps) for a T15 session, in the same order as data.load_sentences."""
    import h5py
    _, path, _, ms = data.SENTENCE_SETS[name]
    out = []
    with h5py.File(path) as f:
        for k in sorted(f.keys()):
            s = f[k].attrs["sentence_label"]
            out.append((f[k]["input_features"][:].astype(np.float32), (s.decode() if isinstance(s, bytes) else str(s)).strip()))
    return out, ms / 1000


def binned(x: np.ndarray, n: int) -> np.ndarray:
    t = (len(x) // n) * n
    return x[:t].reshape(-1, n, x.shape[1]).mean(1)


def main(name: str = "t15_small", nq: int = 48, k: int = 192, seeds=(0, 1)) -> None:
    from sklearn.decomposition import PCA
    trials, step = raw_trials(name)
    order = np.random.default_rng(0).permutation(len(trials))     # the benchmark's split
    used = [trials[i] for i in np.r_[order[:nq], order[nq:nq + k]]]
    secs = [len(x) * step for x, _ in used]

    def split(arrs):                                              # -> (context, queries) in baselines' format
        items = [(a, used[i][1], secs[i]) for i, a in enumerate(arrs)]
        return items[nq:], items[:nq]

    def reduce(nbin: int, pcs: int | None, deciles: bool):
        xs = [binned(x, nbin) for x, _ in used]
        allx = np.concatenate(xs)
        if pcs:
            z = PCA(pcs, random_state=0).fit_transform(allx)
        else:
            z = allx
        if deciles:
            edges = np.quantile(z, np.linspace(0.1, 0.9, 9), axis=0)
            z = (z[:, None, :] > edges[None]).sum(1).astype(np.float32)
        else:
            z = (z - z.mean(0)) / (z.std(0) + 1e-6)
            z = (np.clip(z, -3, 3) + 1) * 4.5                     # same numeric range the decoder expects for deciles
        return np.split(z, np.cumsum([len(x) for x in xs])[:-1])

    # 1) variance that survives each step, relative to the 20 ms recording
    raw = np.concatenate([x[: (len(x) // 8) * 8] for x, _ in used])
    b = np.concatenate([binned(x, 8) for x, _ in used])
    total = raw.var(0).sum()
    after_time = b.var(0).sum() / total
    pca = PCA(12, random_state=0).fit(b)
    z = pca.transform(b)
    after_pca = after_time * pca.explained_variance_ratio_.sum()
    edges = np.quantile(z, np.linspace(0.1, 0.9, 9), axis=0)
    q = (z[:, None, :] > edges[None]).sum(1)
    r2 = np.array([np.corrcoef(z[:, j], q[:, j])[0, 1] ** 2 for j in range(12)])
    after_dec = after_time * (pca.explained_variance_ * r2).sum() / pca.explained_variance_.sum() * pca.explained_variance_ratio_.sum()
    signal = dict(after_time_averaging=float(after_time), after_12_components=float(after_pca), after_deciles=float(after_dec),
                  decile_r2_mean=float(r2.mean()))
    print(f"variance kept, relative to the 20 ms recording: after 160 ms averaging {after_time:.1%}, after 12 components {after_pca:.1%}, "
          f"after deciles {after_dec:.1%} (deciles keep {r2.mean():.1%} of each component)", flush=True)

    # 2) what a fixed decoder gets from each representation
    stages = [("all 512 features, 160 ms frames", 8, None, False),
              ("40 components, 160 ms, continuous", 8, 40, False),
              ("12 components, 160 ms, continuous", 8, 12, False),
              ("12 components, 160 ms, deciles (what the models are given)", 8, 12, True),
              ("24 components, 160 ms, deciles (2x the tokens)", 8, 24, True),
              ("12 components, 80 ms, deciles (2x the tokens)", 4, 12, True)]
    rows = []
    for label, nbin, pcs, dec in stages:
        ctx, qs = split(reduce(nbin, pcs, dec))
        wers, cers = [], []
        for s in seeds:
            preds = ctc(ctx, qs, seed=s)
            sc = score(preds, [x[1] for x in qs], [x[2] for x in qs], n_perm=200)
            wers.append(sc["wer"]); cers.append(sc["cer"])
        rows.append(dict(representation=label, wer=float(np.mean(wers)), cer=float(np.mean(cers)), wer_seeds=wers, cer_seeds=cers))
        print(f"{label:62s} WER {np.mean(wers):.3f} ({', '.join(f'{w:.2f}' for w in wers)})  CER {np.mean(cers):.3f}", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / f"results/lossiness_{name}.json").write_text(json.dumps(dict(dataset=name, signal=signal, decoder=rows), indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:2])
