"""
Предсказание на test и формирование файла сабмита.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    USE_LOG_TARGET,
    MODELS_DIR,
    SUBMISSIONS_DIR,
    TARGET_COL,
    ID_COL,
    ensure_dirs,
)
from .data import load_and_merge, get_test_ids
from .features import prepare_features


def load_model(path: Path) -> tuple[object, list[str]]:
    """Загружает сохранённую модель и список признаков."""
    import joblib

    data = joblib.load(path)
    return data["model"], data["feature_names"]


def predict(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """
    Предсказание. X должен содержать колонки feature_names.
    Если признаков не хватает — дополняем нулями.
    """
    X = X.copy()
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_names]
    return model.predict(X)


def run_predict(
    model_path: Path | None = None,
    model_name: str = "model",
    raw_dir: Path | None = None,
    with_macro: bool = True,
    submission_name: str = "submission",
    **feat_kwargs,
) -> pd.DataFrame:
    """
    Загружает модель и данные, готовит фичи, предсказывает, сохраняет сабмит.
    Возвращает DataFrame с колонками id, price_doc.
    """
    from .config import RAW_DIR

    ensure_dirs()
    raw_dir = raw_dir or RAW_DIR
    model_path = model_path or MODELS_DIR / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    model, feature_names = load_model(model_path)
    train, test = load_and_merge(raw_dir, with_macro=with_macro)
    _, X_test, _ = prepare_features(train, test, **feat_kwargs)

    pred = predict(model, X_test, feature_names)
    if USE_LOG_TARGET:
        pred = np.expm1(pred)

    ids = get_test_ids(test)
    submission = pd.DataFrame({ID_COL: ids, TARGET_COL: pred})

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SUBMISSIONS_DIR / f"{submission_name}.csv"
    submission.to_csv(out_path, index=False)
    return submission
