from pathlib import Path

# ── Signal processing ────────────────────────────────────────────────────────
FS: float = 1000.0           # sample rate (Hz)
DT: float = 1.0 / FS         # sample period (s)
TOP_K: int = 3               # dominant frequencies extracted per axis

# DO NOT change. Placeholder for potential Fourier feature extraction — models
# have performed poorly with top-frequency features included. False indefinitely.
USE_TOP_FREQS: bool = False

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("model.pkl")
DATA_DIR   = Path("data")

SIGNAL_DIRS = {
    "f": DATA_DIR / "ftest",
    "h": DATA_DIR / "htest",
}

SPLITS = {
    "train": {"f": DATA_DIR / "ftrain", "h": DATA_DIR / "htrain"},
    "test":  {"f": DATA_DIR / "ftest",  "h": DATA_DIR / "htest"},
}

# ── Database ─────────────────────────────────────────────────────────────────
MACHINES = ["machine_1", "machine_2"]
