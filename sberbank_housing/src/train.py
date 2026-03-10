"""
Обучение модели: разбиение, обучение, сохранение модели и метрик.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    TARGET_COL,
    USE_LOG_TARGET,
    RANDOM_STATE,
    TEST_SIZE,
    MODELS_DIR,
    XGB_PARAMS,
    LGBM_PARAMS,
    TIMESTAMP_COL,
    ensure_dirs,
)
from .data import load_and_merge
from .features import prepare_features


def get_x_y(train: pd.DataFrame, test: pd.DataFrame, **feat_kwargs) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Возвращает X_train, y_train, X_test, feature_names."""
    y_train = train[TARGET_COL].copy()
    if USE_LOG_TARGET:
        y_train = np.log1p(y_train)

    X_train, X_test, feature_names = prepare_features(train, test, **feat_kwargs)
    return X_train, y_train, X_test, feature_names


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamp: pd.Series,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разбиение по времени: последние test_size процентов по дате — валидация.
    Требует, чтобы X и timestamp были выровнены по индексу.
    """
    idx = timestamp.sort_values().index
    n = int(len(idx) * (1 - test_size))
    train_idx, val_idx = idx[:n], idx[n:]
    return X.loc[train_idx], X.loc[val_idx], y.loc[train_idx], y.loc[val_idx]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE. Если целевая в логе — это RMSE в лог-пространстве."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
) -> "xgb.XGBRegressor":
    """Обучает XGBoost и возвращает модель."""
    import xgboost as xgb

    params = {**XGB_PARAMS, **(params or {})}
    model = xgb.XGBRegressor(**params)
    eval_set = [(X_train, y_train)]
    if X_val is not None and y_val is not None:
        eval_set.append((X_val, y_val))
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=100,
    )
    return model


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict | None = None,
) -> "lightgbm.LGBMRegressor":
    """Обучает LightGBM и возвращает модель."""
    import lightgbm as lgb

    params = {**LGBM_PARAMS, **(params or {})}
    model = lgb.LGBMRegressor(**params)
    callbacks = [lgb.log_evaluation(period=100)]
    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
    else:
        model.fit(X_train, y_train, callbacks=callbacks)
    return model


def run_train(
    raw_dir: Path | None = None,
    with_macro: bool = True,
    model_type: str = "lgbm",  # "lgbm" | "xgb"
    use_time_split: bool = True,
    save_model: bool = True,
    model_name: str = "model",
    **feat_kwargs,
) -> tuple[object, pd.DataFrame, pd.DataFrame, list[str], dict]:
    """
    Полный цикл: загрузка → фичи → разбиение → обучение → сохранение.
    Возвращает (model, X_train, X_test, feature_names, metrics).
    """
    from .config import RAW_DIR

    ensure_dirs()
    raw_dir = raw_dir or RAW_DIR

    train, test = load_and_merge(raw_dir, with_macro=with_macro)
    X_train, y_train, X_test, feature_names = get_x_y(train, test, **feat_kwargs)

    if use_time_split and TIMESTAMP_COL in train.columns:
        ts = train.loc[X_train.index, TIMESTAMP_COL]
        X_tr, X_val, y_tr, y_val = time_based_split(X_train, y_train, ts, test_size=TEST_SIZE)
        eval_X, eval_y = X_val, y_val
    else:
        from sklearn.model_selection import train_test_split

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        eval_X, eval_y = X_val, y_val

    if model_type == "xgb":
        model = train_xgb(X_tr, y_tr, X_val=eval_X, y_val=eval_y)
    else:
        model = train_lgbm(X_tr, y_tr, X_val=eval_X, y_val=eval_y)

    pred_val = model.predict(eval_X)
    val_rmse = rmse(eval_y.values, pred_val)
    metrics = {"val_rmse": val_rmse}

    if save_model:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        import joblib

        path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump({"model": model, "feature_names": feature_names}, path)
        metrics["model_path"] = str(path)

    return model, X_train, X_test, feature_names, metrics
