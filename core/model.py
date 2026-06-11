import pickle

import numpy as np

from constants import MODEL_PATH

_bundle: dict | None = None


def _load_bundle() -> dict:
    global _bundle
    if _bundle is None:
        with MODEL_PATH.open("rb") as f:
            _bundle = pickle.load(f)
    return _bundle


def get_use_top_freqs() -> bool:
    return bool(_load_bundle().get("use_top_freqs", False))


def predict(features: dict[str, float]) -> dict:
    bundle = _load_bundle()
    model = bundle["model"]
    feature_cols: list[str] = bundle["feature_cols"]

    X = np.array([[features[col] for col in feature_cols]])
    prediction = str(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    prob_map = dict(zip(model.classes_, probabilities))

    return {
        "prediction": prediction,
        "probability_faulty": float(prob_map.get("f", 0.0)),
        "probability_healthy": float(prob_map.get("h", 0.0)),
    }
