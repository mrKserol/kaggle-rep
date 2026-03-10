"""
Конфигурация путей и параметров пайплайна.
"""
from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Данные
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Артефакты
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# Имена файлов
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
MACRO_CSV = "macro.csv"
SAMPLE_SUBMISSION_CSV = "sample_submission.csv"

# Целевая переменная и идентификатор
TARGET_COL = "price_doc"
ID_COL = "id"
TIMESTAMP_COL = "timestamp"

# Лог-трансформация целевой 
USE_LOG_TARGET = True

# Параметры обучения 
RANDOM_STATE = 42
TEST_SIZE = 0.2  # доля валидации при разбиении по времени
N_FOLDS = 5      # для cross-validation 

# XGBoost / LightGBM — дефолтные параметры
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


def ensure_dirs():
    """Создаёт нужные директории, если их нет."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, SUBMISSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
