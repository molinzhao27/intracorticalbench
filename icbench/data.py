"""Datasets and the label-free feature pipeline that turns neural activity into model input.

Everything here is unsupervised (binning, PCA, decile quantisation): no sentence or word label
touches the features, so whatever decoding happens has to happen inside the model.
"""
from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Folder holding the downloaded public datasets (see README, "Data"). Set ICB_RAW_DATA in .env.
BIT = Path(os.environ.get("ICB_RAW_DATA", ROOT / "data/raw"))

# name -> (loader kind, path, frames averaged per output frame, source frame ms)
SENTENCE_SETS = {
    # Card et al. 2024 (Brain-to-Text '25), participant T15. First session: 50-word vocabulary.
    "t15_small": ("t15", ROOT / "data/t15/t15.2023.08.11/data_train.hdf5", 8, 20),
    # T15, open vocabulary (Switchboard-style sentences).
    "t15_open": ("t15", ROOT / "data/t15/t15.2023.09.03/data_train.hdf5", 8, 20),
    # Willett et al. 2023, participant T12, open vocabulary.
    "t12_open": ("t12", BIT / "raw/willett_2023_text/train/t12.2022.07.21.mat", 5, 20),
    # More T15 days, open vocabulary: day 2, and sessions 8 and 17 months after the first.
    "t15_day2": ("t15", ROOT / "data/t15/t15.2023.08.13/data_train.hdf5", 8, 20),
    "t15_sep01": ("t15", ROOT / "data/t15/t15.2023.09.01/data_train.hdf5", 8, 20),
    "t15_2024": ("t15", ROOT / "data/t15/t15.2024.04.25/data_train.hdf5", 8, 20),
    "t15_2025": ("t15", ROOT / "data/t15/t15.2025.01.12/data_train.hdf5", 8, 20),
    # Kunz et al. 2025, 50-word vocabulary: another participant attempting speech, and inner speech.
    "t16_small": ("kunz", ROOT / "data/kunz/t16.2024.04.29_attempted_50words.mat", 6, 20),
    "t12_inner": ("kunz", ROOT / "data/kunz/t12.2023.11.28_imagined_50words.mat", 6, 20),
    "t15_inner": ("kunz", ROOT / "data/kunz/t15.2024.03.03_imagined_50words.mat", 16, 10),
}
# What the participant was doing, for the prompt. Default: attempting to say the sentence aloud.
BEHAVIOUR = {"t12_inner": "imagine saying it (inner speech, no movement or sound)",
             "t15_inner": "imagine saying it (inner speech, no movement or sound)"}
WORD_SETS = {p: BIT / f"raw/isolatedVerbalBehaviors/{s}_attempted_raw.mat" for p, s in {
    "t12": "t12.2023.08.15", "t15": "t15.2024.04.07", "t16": "t16.2024.03.04", "t17": "t17.2024.12.09"}.items()}


@dataclass
class Trial:
    x: np.ndarray          # (frames, features) binned activity, before PCA
    label: str             # sentence or word
    seconds: float


def _bin(x: np.ndarray, n: int) -> np.ndarray:
    t = (len(x) // n) * n
    return x[:t].reshape(-1, n, x.shape[1]).mean(1)


def load_sentences(name: str, frame: int | None = None) -> tuple[list[Trial], float]:
    """frame: source steps averaged per output frame; default is the dataset's own setting."""
    kind, path, nbin, ms = SENTENCE_SETS[name]
    nbin = frame or nbin
    out = []
    if kind == "t15":
        import h5py
        with h5py.File(path) as f:
            for k in sorted(f.keys()):
                s = f[k].attrs["sentence_label"]
                s = s.decode() if isinstance(s, bytes) else str(s)
                x = f[k]["input_features"][:].astype(np.float32)   # already z-scored by the authors
                out.append(Trial(_bin(x, nbin), s.strip(), len(x) * ms / 1000))
    elif kind == "kunz":
        import scipy.io as sio
        m = sio.loadmat(path)
        x = np.concatenate([m["binnedTX"], m["spikePow"]], 1).astype(np.float32)
        blocks = m["blockNum"].ravel()
        for b in np.unique(blocks):                              # per-block z-score, no labels involved
            i = blocks == b
            x[i] = (x[i] - x[i].mean(0)) / (x[i].std(0) + 1e-6)
        for (g0, g1), s in zip(m["goTrialEpochs"], m["sentenceText"].ravel()):
            seg = x[int(g0):int(g1)]
            out.append(Trial(_bin(seg, nbin), str(np.ravel(s)[0]).strip(), len(seg) * ms / 1000))
    else:
        import scipy.io as sio
        m = sio.loadmat(path)
        raw = [np.concatenate([m["tx1"][0, i][:, :128], m["spikePow"][0, i][:, :128]], 1).astype(np.float32)
               for i in range(len(m["sentenceText"]))]            # area 6v only (first 128 electrodes)
        allx = np.concatenate(raw)
        mu, sd = allx.mean(0), allx.std(0) + 1e-6
        for x, s in zip(raw, m["sentenceText"]):
            out.append(Trial(_bin((x - mu) / sd, nbin), str(s).strip(), len(x) * ms / 1000))
    return out, nbin * ms / 1000


def load_words(participant: str) -> list[Trial]:
    """Kunz et al. 2025 isolated attempted words (7 words). One time-averaged vector per trial."""
    import scipy.io as sio
    m = sio.loadmat(WORD_SETS[participant])
    x = np.concatenate([m["binnedTX"], m["spikePow"]], 1).astype(np.float32)
    blocks = m["blockNum"].ravel()
    for b in np.unique(blocks):                                  # per-block z-score removes drift
        i = blocks == b
        x[i] = (x[i] - x[i].mean(0)) / (x[i].std(0) + 1e-6)
    cues = [str(c[0]) for c in m["cueList"].ravel()]
    n = int(1500 / int(m["binSize"].ravel()[0]))                 # first 1.5 s after the go cue
    out = []
    for (g0, _), c in zip(m["goTrialEpochs"], m["trialCues"].ravel()):
        word = cues[int(c) - 1]
        if word != "DO_NOTHING":
            out.append(Trial(x[int(g0):int(g0) + n].mean(0, keepdims=True), word, 1.5))
    return out


def quantise(trials: list[Trial], k: int, levels: int = 10) -> list[np.ndarray]:
    """PCA to k components (fit on all given trials' frames, no labels), then per-component quantile
    levels (10 = deciles)."""
    from sklearn.decomposition import PCA
    frames = np.concatenate([t.x for t in trials])
    z = PCA(k, random_state=0).fit_transform(frames)
    cut = np.linspace(0.1, 0.9, 9) if levels == 10 else np.linspace(0, 1, levels + 1)[1:-1]
    edges = np.quantile(z, cut, axis=0)                           # (levels - 1, k)
    q = (z[:, None, :] > edges[None]).sum(1).astype(np.int8)      # 0 .. levels - 1
    cuts = np.cumsum([len(t.x) for t in trials])[:-1]
    return np.split(q, cuts)


def plant(trials: list[Trial], k: int, unit: str, noise: float = 0.6, seed: int = 0) -> list[np.ndarray]:
    """Positive control: same sentences and trial lengths, but the 'activity' is a synthetic code.

    unit="word": every word has one fixed k-dim pattern. unit="letter": every letter has one, so
    unseen words are decodable only by composing letters (the analogue of phonemes).
    """
    rng = np.random.default_rng(seed)
    proto: dict[str, np.ndarray] = {}
    zs = []
    for t in trials:
        n = len(t.x)
        words = normalise(t.label).split()
        z = rng.normal(0, noise, (n, k))
        lo, hi = int(0.1 * n), int(0.9 * n)
        w = np.array([len(x) + 1 for x in words], float)
        bounds = lo + np.round(np.cumsum(np.r_[0, w]) / w.sum() * (hi - lo)).astype(int)
        for word, a, b in zip(words, bounds[:-1], bounds[1:]):
            b = max(b - 1, a + 1)                                 # one silent frame between words
            units = [word] if unit == "word" else list(word.replace("'", ""))
            for u, seg in zip(units, np.array_split(np.arange(a, b), len(units))):
                if u not in proto:
                    proto[u] = rng.normal(0, 1, k)
                z[seg] += proto[u]
        zs.append(z)
    allz = np.concatenate(zs)
    edges = np.quantile(allz, np.linspace(0.1, 0.9, 9), axis=0)
    return [(z[:, None, :] > edges[None]).sum(1).astype(np.int8) for z in zs]


def as_text(q: np.ndarray) -> str:
    """One line per time frame, digits space-separated (one token each)."""
    return "\n".join(" ".join(map(str, row)) for row in q)


def as_png(q: np.ndarray, cell: int = 14) -> str:
    """Heat map: time left to right, components top to bottom. Returns base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import cm
    from PIL import Image
    rgb = (cm.viridis(q.T / 9.0)[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb).resize((q.shape[0] * cell, q.shape[1] * cell), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def normalise(s: str) -> str:
    s = re.sub(r"[^a-z' ]+", " ", s.lower().replace("’", "'"))
    return re.sub(r"\s+", " ", s).strip()
