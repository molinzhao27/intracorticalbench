"""Classical decoders given exactly what the language model is given: the same quantised
components, the same labelled trials, the same queries. They show how much is decodable from
that input at all, which is what makes a model result (positive or null) interpretable.

  uv run python -m icbench.baselines
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import data
from .run import RESULTS, prepare
from .score import score


def words(a, participant: str) -> dict:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    q, ctx, _, _, _ = prepare(a, "words", participant, 1)
    X, y = np.array([c[0][0] for c in ctx], float), [c[1] for c in ctx]
    Xq, yq = np.array([c[0][0] for c in q], float), [c[1] for c in q]
    cents = {w: X[[l == w for l in y]].mean(0) for w in sorted(set(y))}
    nc = [min(cents, key=lambda w: np.linalg.norm(x - cents[w])) for x in Xq]
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y).predict(Xq)
    blocks = len(set(y))
    return dict(participant=participant, n=len(yq), n_context=len(y), classes=blocks,
                nearest_centroid=float(np.mean([p == t for p, t in zip(nc, yq)])),
                lda=float(np.mean([p == t for p, t in zip(lda, yq)])))


def ctc(ctx, queries, unit: str = "char", epochs: int = 400, seed: int = 0) -> list[str]:
    """Small bidirectional GRU with a CTC loss over characters or whole words, trained from scratch
    on the context trials. The word version can only output words seen in the labelled sentences."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    if unit == "char":
        chars, split, join = list(" 'abcdefghijklmnopqrstuvwxyz"), list, "".join
    else:
        chars, split, join = sorted({w for c in ctx for w in data.normalise(c[1]).split()}), str.split, " ".join
    enc = lambda s: torch.tensor([chars.index(c) + 1 for c in split(data.normalise(s))])
    X = [torch.tensor(c[0] / 4.5 - 1.0, dtype=torch.float32) for c in ctx]
    Y = [enc(c[1]) for c in ctx]
    pad = lambda xs: nn.utils.rnn.pad_sequence(xs, batch_first=True)

    class Net(nn.Module):
        def __init__(self, k):
            super().__init__()
            self.gru = nn.GRU(k, 96, batch_first=True, bidirectional=True)
            self.drop, self.out = nn.Dropout(0.3), nn.Linear(192, len(chars) + 1)

        def forward(self, x):
            return self.out(self.drop(self.gru(x)[0])).log_softmax(-1)

    net = Net(X[0].shape[1])
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3)
    lossf = nn.CTCLoss(zero_infinity=True)
    xl, yl = torch.tensor([len(x) for x in X]), torch.tensor([len(y) for y in Y])
    xp, yp = pad(X), pad(Y)
    for _ in range(epochs):
        net.train()
        noisy = xp + 0.15 * torch.randn_like(xp)
        loss = lossf(net(noisy).transpose(0, 1), yp, xl, yl)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    out = []
    with torch.no_grad():
        for c in queries:
            ids = net(torch.tensor(c[0] / 4.5 - 1.0, dtype=torch.float32)[None])[0].argmax(-1).tolist()
            dec = [i for i, j in zip(ids, [0] + ids[:-1]) if i != j and i != 0]
            out.append(join(chars[i - 1] for i in dec))
    return out


def sentences(a, dataset: str, k: int) -> list[dict]:
    q, ctx, _, _, _ = prepare(a, "sentences", dataset, k)
    refs, secs = [x[1] for x in q], [x[2] for x in q]
    feat = lambda x: np.r_[x[0].mean(0), x[0].std(0), len(x[0]) / 10.0]
    C = np.array([feat(c) for c in ctx])
    nn_pred = [ctx[int(np.argmin(np.linalg.norm(C - feat(x), axis=1)))][1] for x in q]
    rows = []
    for name, preds in (("nearest labelled trial", nn_pred), ("GRU-CTC, characters", ctc(ctx, q)), ("GRU-CTC, words", ctc(ctx, q, "word"))):
        rows.append(dict(dataset=dataset, k=k, decoder=name, preds=preds, refs=refs, score=score(preds, refs, secs)))
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sentences", nargs="*", default=["t15_small", "t15_open", "planted_word:t15_small", "planted_letter:t15_small"])
    p.add_argument("--words", nargs="*", default=["t12", "t15", "t16", "t17"])
    p.add_argument("--k", nargs="+", type=int, default=[16, 64, 192])
    p.add_argument("--pcs", type=int, default=12)
    p.add_argument("--nq", type=int, default=48)
    p.add_argument("--nq-per-word", type=int, default=6)
    p.add_argument("--kmax", type=int, default=192)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = {"words": [words(a, w) for w in a.words], "sentences": []}
    for w in out["words"]:
        print(f"words {w['participant']}: nearest centroid {w['nearest_centroid']:.3f}, LDA {w['lda']:.3f} (n={w['n']}, context {w['n_context']})")
    for ds in a.sentences:
        for k in a.k:
            for r in sentences(a, ds, k):
                s = r["score"]
                print(f"{ds} k={k} {r['decoder']}: WER {s['wer']:.3f} (duration-matched chance {s['wer_chance_len']:.3f}, p={s['wer_p_len']:.4f})  "
                      f"CER {s['cer']:.3f} (chance {s['cer_chance_len']:.3f}, p={s['cer_p_len']:.4f})", flush=True)
                out["sentences"].append(r)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS.parent / "baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
