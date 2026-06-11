import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


FS = 1000.0
DT = 1.0 / FS
N_SPLITS = 5
TEST_FRACTION = 0.2

DATA_DIR = Path("data")
ARCHIVE_DIR = DATA_DIR / "archive"
SOURCE_DIRS = {"h": ARCHIVE_DIR / "Healthy", "f": ARCHIVE_DIR / "Faulty"}
SPLIT_DIRS = {
    ("h", "train"): DATA_DIR / "htrain",
    ("h", "test"): DATA_DIR / "htest",
    ("f", "train"): DATA_DIR / "ftrain",
    ("f", "test"): DATA_DIR / "ftest",
}


def split_mat_to_csvs(mat_path: Path, out_dir: Path, n_splits: int = N_SPLITS, fs: float = FS):
    data = loadmat(mat_path)["H"]
    df = pd.DataFrame(data, columns=["x", "y", "z"])
    df.insert(0, "t", np.arange(len(df)) / fs)

    base = mat_path.stem
    n = len(df)
    edges = np.linspace(0, n, n_splits + 1, dtype=int)
    for i in range(n_splits):
        chunk = df.iloc[edges[i]:edges[i + 1]]
        chunk.to_csv(out_dir / f"{base}_{i + 1}.csv", index=False)


def main():
    for out_dir in SPLIT_DIRS.values():
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

    for label, src_dir in SOURCE_DIRS.items():
        mat_files = sorted(src_dir.glob("*.mat"))
        random.shuffle(mat_files)
        n_test = int(round(len(mat_files) * TEST_FRACTION))
        test_files = set(mat_files[:n_test])

        for mat_path in mat_files:
            split = "test" if mat_path in test_files else "train"
            out_dir = SPLIT_DIRS[(label, split)]
            split_mat_to_csvs(mat_path, out_dir)

        print(f"{label}: {len(mat_files) - n_test} train / {n_test} test .mat files")


if __name__ == "__main__":
    main()
