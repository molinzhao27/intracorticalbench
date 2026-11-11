"""What kind of mistakes are the wrong words: sound-alikes, meaning-alikes, or neither?

Works on saved predictions only (no model calls), 50-word session `t15_small`. Each prediction is
aligned to its reference word by word; every substitution (said word -> decoded word) is an event.

Three views of the same events, for several groups of runs:

1. Rank. Among the 49 other vocabulary words, how close is the decoded word to the said word in
   sound (articulatory-feature edit distance between phoneme strings, CMUdict + panphon) and in
   meaning (GloVe cosine)? Reported as the share of events in the 5 closest words.
2. Choice model. A conditional logit over the 49 candidates with sound, meaning, phoneme-count
   match, word frequency in the labelled sentences, and fit to the neighbouring decoded words
   (a word pair seen in the labelled sentences) as predictors, all on one scale.
3. Whole sentences. Phoneme error rate and BERTScore of the wrong sentences.

The nulls in 1 and 3 keep everything but the identity of the wrong words. "frequency": each one is
redrawn from the words that group of runs produces as substitutions. "plausible": redrawn in
proportion to frequency in the labelled sentences and fit to the neighbouring decoded words, so the
random sentence stays about as fluent as the real one (BERTScore rewards fluency). Runs
that had no usable neural information (models at chance, activity withheld) are the other control:
alignment and grammar alone make substitutions look somewhat related.

  uv run --extra analysis python -m icbench.errors            # everything, writes results/errors.json
  uv run --extra analysis python -m icbench.errors --no-bert  # skip BERTScore (no model download)
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter

import numpy as np

from . import agentic
from .data import normalise
from .models import ROOT

RESULTS = ROOT / "runs/results"
ARPA = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɹ̩",
        "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ",
        "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}
FEATURES = ["sound", "meaning", "length", "frequency", "context"]
TOP = 5


def align(ref: list[str], hyp: list[str]) -> list[tuple[str, int, int]]:
    """Levenshtein alignment: ("match" | "sub" | "del" | "ins", index in ref, index in hyp)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("match" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", i - 1, j))
            i -= 1
        else:
            ops.append(("ins", i, j - 1))
            j -= 1
    return ops[::-1]


def lev(a: list, b: list, cost=lambda x, y: float(x != y)) -> float:
    """Edit distance; insertions and deletions cost 1, a substitution costs `cost(x, y)` in [0, 1]."""
    prev = [float(j) for j in range(len(b) + 1)]
    for i, x in enumerate(a, 1):
        cur = [float(i)]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + cost(x, y)))
        prev = cur
    return prev[-1]


class Words:
    """The session's vocabulary with pairwise sound and meaning similarity (z-scored over all pairs)."""

    def __init__(self, labelled: list[str], held_out: list[str]):
        import cmudict
        import torch
        from huggingface_hub import hf_hub_download
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import panphon
        sents = [normalise(s).split() for s in labelled]
        self.vocab = sorted({w for s in sents + [normalise(s).split() for s in held_out] for w in s})
        self.ix = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        cmu = cmudict.dict()
        self.cmu = cmu
        self.phones = {w: [p.rstrip("012") for p in cmu[w][0]] for w in self.vocab}
        # each phoneme as articulatory feature vectors (diphthongs and affricates are two segments)
        ft = panphon.FeatureTable()
        self.segs = {w: [tuple(v) for p in ph for v in ft.word_to_vector_list(ARPA[p], numeric=True)] for w, ph in self.phones.items()}
        seg_cost = lambda x, y: sum(abs(i - j) for i, j in zip(x, y)) / (2 * len(x))   # share of features that differ
        repo = "sentence-transformers/average_word_embeddings_glove.6B.300d"
        gv = json.load(open(hf_hub_download(repo, "0_WordEmbeddings/whitespacetokenizer_config.json")))["vocab"]
        emb = torch.load(hf_hub_download(repo, "0_WordEmbeddings/pytorch_model.bin"), map_location="cpu")["emb_layer.weight"].numpy()
        g = {w: i for i, w in enumerate(gv)}
        vec = np.stack([emb[g[w]] for w in self.vocab])
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        self.raw = dict(sound=np.zeros((V, V)), sound_plain=np.zeros((V, V)), meaning=vec @ vec.T, length=np.zeros((V, V)))
        for a in self.vocab:
            for b in self.vocab:
                i, j = self.ix[a], self.ix[b]
                self.raw["sound"][i, j] = -lev(self.segs[a], self.segs[b], seg_cost) / max(len(self.segs[a]), len(self.segs[b]))
                self.raw["sound_plain"][i, j] = -lev(self.phones[a], self.phones[b]) / max(len(self.phones[a]), len(self.phones[b]))
                self.raw["length"][i, j] = -abs(len(self.phones[a]) - len(self.phones[b]))
        off = ~np.eye(V, dtype=bool)
        self.z = {k: (m - m[off].mean()) / m[off].std() for k, m in self.raw.items()}
        count = Counter(w for s in sents for w in s)
        f = np.log1p(np.array([count[w] for w in self.vocab], float))
        self.freq = (f - f.mean()) / f.std()
        self.count = count
        self.pairs = {(a, b) for s in sents for a, b in zip(["<s>"] + s, s + ["</s>"])}
        # rank of j among the other words by closeness to i (1 = closest), per measure
        self.rank = {}
        for k in ("sound", "sound_plain", "meaning"):
            m = self.raw[k].copy()
            np.fill_diagonal(m, -np.inf)
            self.rank[k] = (-m).argsort(axis=1).argsort(axis=1) + 1

    def sentence_phones(self, toks: list[str]) -> list[str] | None:
        out = []
        for w in toks:
            if w in self.phones:
                out += self.phones[w]
            elif w in self.cmu:
                out += [p.rstrip("012") for p in self.cmu[w][0]]
            else:
                return None
        return out


def events(W: Words, cells: list[dict]) -> tuple[list[dict], dict]:
    """Substitution events of a group of cells, and counts of what kinds of edits its errors are."""
    ev, kinds, sent = [], Counter(), 0
    for c in cells:
        for q, (p, r) in enumerate(zip(c["preds"], c["refs"])):
            if not p:
                continue
            ref, hyp = normalise(r).split(), normalise(p).split()
            if ref == hyp or not hyp:
                continue
            sent += 1
            ops = align(ref, hyp)
            kinds.update(o for o, _, _ in ops if o != "match")
            for o, i, j in ops:
                if o == "sub":
                    ev.append(dict(cell=c["name"], q=q, ref=ref[i], hyp=hyp[j], pos=j, toks=hyp, reftoks=ref,
                                   prev=hyp[j - 1] if j else "<s>", next=hyp[j + 1] if j + 1 < len(hyp) else "</s>"))
    n_sub = len(ev)
    ev = [e for e in ev if e["hyp"] in W.ix and e["ref"] in W.ix]
    return ev, dict(wrong_sentences=sent, substitutions=n_sub, deletions=kinds["del"], insertions=kinds["ins"], outside_vocabulary=n_sub - len(ev))


def replacement_probs(W: Words, ev: list[dict]) -> np.ndarray:
    """Per event, the distribution a random wrong word is drawn from: this group's substituted-in words, minus the said word."""
    marg = np.zeros(len(W.vocab))
    for e in ev:
        marg[W.ix[e["hyp"]]] += 1
    P = np.tile(marg, (len(ev), 1))
    for n, e in enumerate(ev):
        P[n, W.ix[e["ref"]]] = 0
    return P / P.sum(1, keepdims=True)


def plausible_probs(W: Words, ev: list[dict]) -> np.ndarray:
    """A stricter random wrong word: drawn in proportion to how often the word occurs in the labelled sentences and how
    well it fits the neighbouring decoded words (a choice model with only those two predictors, fitted to this group)."""
    X, y, mask = design(W, ev)
    keep = [FEATURES.index("frequency"), FEATURES.index("context")]
    u = np.where(mask, X[:, :, keep] @ fit_logit(X[:, :, keep], y, mask), -np.inf)
    u -= u.max(1, keepdims=True)
    return np.exp(u) / np.exp(u).sum(1, keepdims=True)


def rank_view(W: Words, ev: list[dict], nulls: dict[str, np.ndarray], rng, n_draw: int = 2000) -> dict:
    r = np.array([W.ix[e["ref"]] for e in ev])
    h = np.array([W.ix[e["hyp"]] for e in ev])
    out = {}
    for k in ("sound", "sound_plain", "meaning"):
        top = W.rank[k] <= TOP
        actual = top[r, h].mean()
        out[k] = dict(top5=round(float(actual), 3))
        for tag, P in nulls.items():
            draws = (rng.random((n_draw, len(ev), 1)) > P.cumsum(1)[None]).sum(2).clip(max=len(W.vocab) - 1)     # n_draw x events
            null = top[r[None], draws].mean(1)
            out[k] |= {f"expected_{tag}": round(float(null.mean()), 3), f"p_{tag}": round(float(((null >= actual).sum() + 1) / (n_draw + 1)), 4)}
    fits = np.array([((e["prev"], e["hyp"]) in W.pairs) or ((e["hyp"], e["next"]) in W.pairs) for e in ev])
    out["fits_neighbours"] = round(float(fits.mean()), 3)
    pairs = sorted({(a, b) for a, b in zip(r, h)})                         # each confusion counted once
    out["distinct_pairs"] = dict(n=len(pairs), sound_top5=round(float(np.mean([W.rank["sound"][a, b] <= TOP for a, b in pairs])), 3),
                                 meaning_top5=round(float(np.mean([W.rank["meaning"][a, b] <= TOP for a, b in pairs])), 3), uniform=round(TOP / (len(W.vocab) - 1), 3))
    return out


def design(W: Words, ev: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """events x candidates x features, and the index of the chosen candidate. The said word is not a candidate."""
    V = len(W.vocab)
    X = np.zeros((len(ev), V, len(FEATURES)))
    for n, e in enumerate(ev):
        r = W.ix[e["ref"]]
        X[n, :, 0], X[n, :, 1], X[n, :, 2], X[n, :, 3] = W.z["sound"][r], W.z["meaning"][r], W.z["length"][r], W.freq
        X[n, :, 4] = [((e["prev"], w) in W.pairs) + ((w, e["next"]) in W.pairs) for w in W.vocab]
    mask = np.ones((len(ev), V), bool)
    mask[np.arange(len(ev)), [W.ix[e["ref"]] for e in ev]] = False
    X[~mask] = 0
    return X, np.array([W.ix[e["hyp"]] for e in ev]), mask


def fit_logit(X: np.ndarray, y: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    from scipy.optimize import minimize
    from scipy.special import logsumexp
    w = np.ones(len(y)) if weights is None else weights
    rows = np.arange(len(y))

    def nll(b):
        u = np.where(mask, X @ b, -np.inf)
        lse = logsumexp(u, axis=1)
        p = np.exp(u - lse[:, None])
        grad = -(w[:, None] * (X[rows, y] - (p[:, :, None] * X).sum(1))).sum(0) + 2e-3 * b
        return -(w * (u[rows, y] - lse)).sum() + 1e-3 * (b @ b), grad

    return minimize(nll, np.zeros(X.shape[2]), jac=True, method="L-BFGS-B").x


def choice_view(W: Words, ev: list[dict], rng, n_boot: int = 300) -> dict:
    X, y, mask = design(W, ev)
    beta = fit_logit(X, y, mask)
    groups = sorted({(e["cell"], e["q"]) for e in ev})                     # resample whole sentences
    gi = {g: i for i, g in enumerate(groups)}
    owner = np.array([gi[(e["cell"], e["q"])] for e in ev])
    boots = []
    for _ in range(n_boot):
        times = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        boots.append(fit_logit(X, y, mask, times[owner].astype(float)))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {f: dict(beta=round(float(b), 3), lo=round(float(l), 3), hi=round(float(h), 3), odds=round(math.exp(b), 2)) for f, b, l, h in zip(FEATURES, beta, lo, hi)}


def sentence_view(W: Words, ev: list[dict], nulls: dict[str, np.ndarray], rng, scorer, n_draw: int = 100, cap: int = 400) -> dict:
    """Wrong sentences against the same sentences with each wrong word redrawn at random, under each null."""
    return {tag: _sentence_view(W, ev, P, rng, scorer, n_draw, cap) for tag, P in nulls.items()}


def _sentence_view(W: Words, ev: list[dict], P: np.ndarray, rng, scorer, n_draw: int, cap: int) -> dict:
    by = {}
    for n, e in enumerate(ev):
        by.setdefault((e["cell"], e["q"]), []).append(n)
    keys = sorted(by)
    if len(keys) > cap:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), cap, replace=False))]
    cum = P.cumsum(1)
    actual, fake, refs = [], [], []
    for key in keys:
        e0 = ev[by[key][0]]
        refs.append(e0["reftoks"])
        actual.append(e0["toks"])
        alts = []
        for _ in range(n_draw):
            t = list(e0["toks"])
            for n in by[key]:
                t[ev[n]["pos"]] = W.vocab[min(int((rng.random() > cum[n]).sum()), len(W.vocab) - 1)]
            alts.append(t)
        fake.append(alts)

    def per(hyp, ref):
        a, b = W.sentence_phones(hyp), W.sentence_phones(ref)
        return None if a is None or b is None else lev(a, b) / len(b)

    def summarise(act: np.ndarray, alt: np.ndarray, lower_is_closer: bool) -> dict:
        ok = ~np.isnan(act) & ~np.isnan(alt).any(1)
        act, alt = act[ok], alt[ok]
        null = alt.mean(0)                                         # one mean per redraw
        closer = (null <= act.mean()).sum() if lower_is_closer else (null >= act.mean()).sum()
        return dict(sentences=int(ok.sum()), actual=round(float(act.mean()), 3), random_words=round(float(null.mean()), 3),
                    z=round(float((act.mean() - null.mean()) / null.std()), 2), p=round(float((closer + 1) / (alt.shape[1] + 1)), 3))

    nan = float("nan")
    out = dict(phoneme_error_rate=summarise(np.array([nan if per(a, r) is None else per(a, r) for a, r in zip(actual, refs)]),
                                            np.array([[nan if per(t, r) is None else per(t, r) for t in alts] for alts, r in zip(fake, refs)]), True))
    if scorer is not None:
        pairs = sorted({(" ".join(t), " ".join(r)) for alts, a, r in zip(fake, actual, refs) for t in alts + [a]})
        f1 = scorer.score([p[0] for p in pairs], [p[1] for p in pairs], batch_size=128)[2].tolist()
        look = dict(zip(pairs, f1))
        out["bertscore_f1"] = summarise(np.array([look[(" ".join(a), " ".join(r))] for a, r in zip(actual, refs)]),
                                        np.array([[look[(" ".join(t), " ".join(r))] for t in alts] for alts, r in zip(fake, refs)]), False)
    return out


def load_groups() -> dict[str, list[dict]]:
    cells = []
    for f in sorted(RESULTS.glob("sentences__t15_small__*.json")):
        c = json.loads(f.read_text())
        if c.get("score") and c["k"] >= 16:
            cells.append(c)
    astra = {"astra", "astra-codex-notools", "astra-codex-notools-trace"}
    in_prompt = lambda c: c["fmt"] == "text" and (c["model"] in ("astra", "fable", "opus", "sonnet", "haiku") or c["model"].endswith(("-notools", "-notools-trace")))
    p = lambda c: c["score"]["wer_p_len"]
    groups = {
        "GPT Astra, in the prompt": [c for c in cells if in_prompt(c) and c["model"] in astra],
        "Other models above chance, in the prompt": [c for c in cells if in_prompt(c) and c["model"] not in astra and p(c) <= 0.01],
        "Models at chance, in the prompt": [c for c in cells if in_prompt(c) and p(c) > 0.05],
        "Activity withheld (guessing)": [c for c in cells if c["fmt"] == "none"],
        "Decoders the models trained (tools)": [c for c in cells if c["fmt"] == "text" and c["model"].endswith(("-codex", "-claudecode"))],
    }
    base = json.loads((ROOT / "results/baselines.json").read_text())["sentences"]
    for name in ("GRU-CTC, words", "nearest labelled trial"):
        groups[f"Classical: {name}"] = [dict(name=f"{name} k{b['k']}", preds=b["preds"], refs=b["refs"]) for b in base if b["dataset"] == "t15_small" and b["decoder"] == name]
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-bert", action="store_true")
    a = ap.parse_args()
    queries, ctx, _, _, _ = agentic.prepare(agentic._args(), "sentences", "t15_small", 192)
    W = Words([x[1] for x in ctx], [x[1] for x in queries])
    scorer = None
    if not a.no_bert:
        import torch
        from bert_score import BERTScorer
        scorer = BERTScorer(lang="en", rescale_with_baseline=True, device="mps" if torch.backends.mps.is_available() else None)
    out = dict(about=dict(
        data="t15_small, cells with 16-192 labelled trials. An event is one substituted word in a word-level alignment of prediction and reference.",
        sound="Edit distance between phoneme strings (CMUdict) where swapping two phonemes costs the share of articulatory features (panphon) that differ, divided by the longer length. sound_plain: every swap costs 1.",
        meaning="Cosine similarity of GloVe 6B 300d vectors.",
        top5="Share of events whose decoded word is among the 5 vocabulary words closest to the said word.",
        nulls="Each wrong word redrawn at random. frequency: from the group's own substituted-in words. plausible: in proportion to the word's frequency in the labelled sentences and its fit to the neighbouring decoded words.",
        choice="Conditional logit over the other vocabulary words; beta per standard deviation of the predictor (context: per neighbouring word pair seen in the labelled sentences). 95% interval from resampling sentences.",
        sentences="Wrong sentences against copies whose wrong words are redrawn at random (100 redraws, up to 400 sentences per group). BERTScore F1: roberta-large, rescaled with baseline."),
        vocabulary=W.vocab, groups={})
    rng = np.random.default_rng(0)
    for name, cells in load_groups().items():
        ev, kinds = events(W, cells)
        nulls = dict(frequency=replacement_probs(W, ev), plausible=plausible_probs(W, ev))
        g = dict(cells=len(cells), events=len(ev), edits=kinds, rank=rank_view(W, ev, nulls, rng), choice=choice_view(W, ev, rng),
                 sentences=sentence_view(W, ev, nulls, rng, scorer),
                 common=[dict(said=r, decoded=h, n=n, sound_rank=int(W.rank["sound"][W.ix[r], W.ix[h]]), meaning_rank=int(W.rank["meaning"][W.ix[r], W.ix[h]]))
                         for (r, h), n in Counter((e["ref"], e["hyp"]) for e in ev).most_common(12)])
        out["groups"][name] = g
        rk, ch, se = g["rank"], g["choice"], g["sentences"]
        print(f"\n{name}: {len(cells)} cells, {len(ev)} substitutions ({kinds})")
        print(f"  top-{TOP} (actual vs frequency null / plausible null): " + " | ".join(
            f"{k} {rk[k]['top5']:.2f} vs {rk[k]['expected_frequency']:.2f} (p={rk[k]['p_frequency']}) / {rk[k]['expected_plausible']:.2f} (p={rk[k]['p_plausible']})"
            for k in ("sound", "sound_plain", "meaning")) + f" | fits neighbours {rk['fits_neighbours']:.2f} | distinct pairs {rk['distinct_pairs']}")
        print("  choice: " + "  ".join(f"{f} {ch[f]['beta']:+.2f} [{ch[f]['lo']:+.2f},{ch[f]['hi']:+.2f}]" for f in FEATURES))
        for tag in se:
            print(f"  sentences vs {tag} null: " + "  ".join(f"{k} {v['actual']:.3f} vs {v['random_words']:.3f} (z={v['z']}, p={v['p']})" for k, v in se[tag].items()))
        print("  common: " + ", ".join(f"{c['said']}->{c['decoded']} x{c['n']} (s{c['sound_rank']}/m{c['meaning_rank']})" for c in g["common"][:8]))
    (ROOT / "results/errors.json").write_text(json.dumps(out, indent=1))
    print("\nwrote results/errors.json")


if __name__ == "__main__":
    main()
