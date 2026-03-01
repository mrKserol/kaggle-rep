import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет feature engineering для датасета House Prices.
    Добавляет новые признаки размера, возраста, ванных комнат и индикаторы наличия объектов.
    """

    df = df.copy()

    # --- Фичи размера ---
    # Общая площадь (подвал + 1й этаж + 2й этаж)
    df["TotalSF"] = (
        df["TotalBsmtSF"] +
        df["1stFlrSF"] +
        df["2ndFlrSF"]
    )

    # --- Возрастные фичи ---
    # Возраст дома на момент продажи (страхуемся от отрицательных значений)
    df["Age"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)

    # Возраст с момента последнего ремонта на момент продажи
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)

    # Бинарный признак: был ли ремонт (год ремонта больше года постройки)
    df["IsRemodeled"] = (df["YearRemodAdd"] > df["YearBuilt"]).astype(int)

    # --- Ванные комнаты ---
    # Полная ванная = 1, половинная = 0.5
    df["TotalBath"] = (
        df["FullBath"] +
        0.5 * df["HalfBath"] +
        df["BsmtFullBath"] +
        0.5 * df["BsmtHalfBath"]
    )

    # --- Индикаторы наличия объектов ---
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)

    # --- Взаимодействие качества и состояния ---
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]

    return df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return feature_engineering(X)

