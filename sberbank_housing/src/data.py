"""
Загрузка и объединение данных: train, test, macro.
"""
from pathlib import Path

import pandas as pd

from .config import (
    RAW_DIR,
    TIMESTAMP_COL,
    TRAIN_CSV,
    TEST_CSV,
    MACRO_CSV,
    TARGET_COL,
    ID_COL,
)


def load_train(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Загружает train.csv."""
    path = raw_dir / TRAIN_CSV
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")
    df = pd.read_csv(path)
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    return df


def load_test(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Загружает test.csv."""
    path = raw_dir / TEST_CSV
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")
    df = pd.read_csv(path)
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    return df


def load_macro(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Загружает macro.csv."""
    path = raw_dir / MACRO_CSV
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")
    df = pd.read_csv(path)
    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    return df


def merge_with_macro(
    df: pd.DataFrame,
    macro: pd.DataFrame,
    on: str = TIMESTAMP_COL,
    how: str = "left",
) -> pd.DataFrame:
    """
    Джойнит таблицу сделок с макро по дате.
    Для каждой даты в df подтягиваются макро-показатели на эту дату.
    """
    # Приводим к дате без времени для однозначного совпадения
    df = df.copy()
    macro = macro.copy()
    date_col = "_date"
    df[date_col] = df[on].dt.normalize()
    macro[date_col] = macro[on].dt.normalize()
    merged = df.merge(
        macro.drop(columns=[on]),
        on=date_col,
        how=how,
        suffixes=("", "_macro"),
    )
    merged = merged.drop(columns=[date_col])
    return merged


def load_and_merge(
    raw_dir: Path = RAW_DIR,
    with_macro: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает train, test и при необходимости объединяет с macro.
    Возвращает (train, test).
    """
    train = load_train(raw_dir)
    test = load_test(raw_dir)

    if with_macro:
        macro = load_macro(raw_dir)
        train = merge_with_macro(train, macro)
        test = merge_with_macro(test, macro)

    return train, test


def get_test_ids(test: pd.DataFrame, id_col: str = ID_COL) -> pd.Series:
    """Возвращает серию id тестовой выборки (для сабмита)."""
    return test[id_col]
