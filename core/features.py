from pathlib import Path

import numpy as np
import pandas as pd

from constants import DT, TOP_K


def _top_freqs(signal: np.ndarray, k: int = TOP_K) -> np.ndarray:
    x = signal - signal.mean()
    N = len(x)
    X = np.fft.rfft(x)
    freq = np.fft.rfftfreq(N, d=DT)
    omega = 2 * np.pi * freq
    A = (2.0 / N) * np.abs(X)
    idx = np.argsort(A)[::-1][:k]
    return omega[idx]


def extract_features(csv_path: Path, use_top_freqs: bool = False) -> dict[str, float]:
    df = pd.read_csv(csv_path)
    features: dict[str, float] = {}
    for axis in ["x", "y", "z"]:
        s = df[axis].to_numpy()
        if use_top_freqs:
            for i, w in enumerate(_top_freqs(s), start=1):
                features[f"{axis}_w{i}"] = float(w)
        features[f"{axis}_max"]  = float(s.max())
        features[f"{axis}_min"]  = float(s.min())
        features[f"{axis}_mean"] = float(s.mean())
    return features
