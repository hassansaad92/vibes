import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from constants import FS, MODEL_PATH, SPLITS, TOP_K, USE_TOP_FREQS
from core.features import extract_features

SAVE_MODEL = True


def build_dataset(split: str) -> pd.DataFrame:
    rows = []
    for label, src_dir in SPLITS[split].items():
        for p in sorted(src_dir.glob("*.csv")):
            features = extract_features(p, use_top_freqs=USE_TOP_FREQS)
            rows.append({"file": p.name, "label": label, **features})
    return pd.DataFrame(rows)


def main():
    print(f"USE_TOP_FREQS = {USE_TOP_FREQS}")

    train_df = build_dataset("train")
    test_df  = build_dataset("test")

    feature_cols = [c for c in train_df.columns if c not in ("file", "label")]
    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test  = test_df[feature_cols].to_numpy()
    y_test  = test_df["label"].to_numpy()

    print(f"features ({len(feature_cols)}): {feature_cols}")
    print("train label counts:", pd.Series(y_train).value_counts().to_dict())
    print("test label counts: ", pd.Series(y_test).value_counts().to_dict())

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print()
    print("f1 (per class):")
    for label, score in zip(["f", "h"], f1_score(y_test, y_pred, labels=["f", "h"], average=None)):
        print(f"  {label}: {score:.4f}")
    print(f"f1 (macro):    {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(f"f1 (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print()
    print(classification_report(y_test, y_pred))
    print("confusion matrix (rows=true [h,f], cols=pred [h,f]):")
    print(confusion_matrix(y_test, y_pred, labels=["h", "f"]))

    if SAVE_MODEL:
        bundle = {
            "model": model,
            "feature_cols": feature_cols,
            "use_top_freqs": USE_TOP_FREQS,
            "fs": FS,
            "top_k": TOP_K,
        }
        with MODEL_PATH.open("wb") as f:
            pickle.dump(bundle, f)
        print(f"\nsaved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
