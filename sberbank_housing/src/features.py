"""
Feature engineering: производные признаки, кодирование, подготовка матриц X/y.
"""
import numpy as np
import pandas as pd

from .config import TARGET_COL, ID_COL, TIMESTAMP_COL


# Признаки, которые всегда исключаем из обучения
DROP_COLS = [ID_COL, TARGET_COL]


def add_date_features(df: pd.DataFrame, date_col: str = TIMESTAMP_COL) -> pd.DataFrame:
    """Добавляет год, месяц, квартал, день недели из даты."""
    df = df.copy()
    if date_col not in df.columns:
        return df
    dt = pd.to_datetime(df[date_col])
    df["_year"] = dt.dt.year
    df["_month"] = dt.dt.month
    df["_quarter"] = dt.dt.quarter
    df["_day_of_week"] = dt.dt.dayofweek
    return df


def add_area_features(df: pd.DataFrame) -> pd.DataFrame:
    """Доли площадей (защита от деления на 0)."""
    df = df.copy()
    if "full_sq" in df.columns and "life_sq" in df.columns:
        df["_life_to_full_sq"] = df["life_sq"] / df["full_sq"].replace(0, np.nan)
    if "full_sq" in df.columns and "kitch_sq" in df.columns:
        df["_kitch_to_full_sq"] = df["kitch_sq"] / df["full_sq"].replace(0, np.nan)
    if "floor" in df.columns and "max_floor" in df.columns:
        df["_floor_to_max"] = df["floor"] / df["max_floor"].replace(0, np.nan)
    return df


def get_numeric_categorical_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Разбивает колонки на числовые и категориальные (object, category)."""
    exclude = exclude or []
    numeric = []
    categorical = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype in ("int64", "float64", "int32", "float32"):
            numeric.append(c)
        elif df[c].dtype in ("object", "category", "bool"):
            categorical.append(c)
    return numeric, categorical


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    add_derived: bool = True,
    fill_na_numeric: str = "median",
    categorical_strategy: str = "drop",  # "drop" | "onehot" | "ordinal"
    drop_high_missing: float = 0.9,  # дропать колонки с долей пропусков > этого
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Готовит матрицы признаков для train и test.
    Возвращает (X_train, X_test, feature_names).
    """
    train = train.copy()
    test = test.copy()

    if add_derived:
        train = add_date_features(train)
        test = add_date_features(test)
        train = add_area_features(train)
        test = add_area_features(test)

    # Убираем колонки без информации
    drop_cols = [c for c in DROP_COLS if c in train.columns]
    train = train.drop(columns=drop_cols, errors="ignore")
    test = test.drop(columns=drop_cols, errors="ignore")

    # Выравниваем колонки: только те, что есть в обоих
    common = list(train.columns.intersection(test.columns))
    train = train[common]
    test = test[common]

    # Дропаем колонки с слишком большим числом пропусков (по train)
    missing_ratio = train.isna().mean()
    cols_keep = missing_ratio[missing_ratio <= drop_high_missing].index.tolist()
    train = train[cols_keep]
    test = test[cols_keep]

    numeric_cols, cat_cols = get_numeric_categorical_columns(train)

    if categorical_strategy == "drop":
        feature_cols = numeric_cols
        train = train[feature_cols]
        test = test[feature_cols]
    elif categorical_strategy == "ordinal":
        # Простое числовое кодирование по train
        feature_cols = numeric_cols.copy()
        for c in cat_cols:
            if c not in train.columns:
                continue
            uniq = train[c].dropna().unique()
            mapping = {v: i for i, v in enumerate(uniq)}
            train[c] = train[c].map(mapping)
            test[c] = test[c].map(mapping)
            feature_cols.append(c)
        train = train[feature_cols]
        test = test[feature_cols]
    else:  # onehot
        train = pd.get_dummies(train, columns=cat_cols, drop_first=True, dtype=float)
        test = pd.get_dummies(test, columns=cat_cols, drop_first=True, dtype=float)
        common = list(train.columns.intersection(test.columns))
        train = train[common]
        test = test[common]
        feature_cols = common

    # Заполнение пропусков в числовых
    if fill_na_numeric == "median":
        med = train.median()
        train = train.fillna(med)
        test = test.fillna(med)
    elif fill_na_numeric == "mean":
        mu = train.mean()
        train = train.fillna(mu)
        test = test.fillna(mu)
    elif fill_na_numeric == "zero":
        train = train.fillna(0)
        test = test.fillna(0)

    # Финальное выравнивание (на случай разного набора one-hot)
    feature_cols = [c for c in feature_cols if c in train.columns and c in test.columns]
    train = train[feature_cols]
    test = test[feature_cols]

    return train, test, feature_cols
