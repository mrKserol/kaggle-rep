#!/usr/bin/env python3
"""
Скрипт запуска пайплайна: обучение и/или предсказание.

Примеры:
  python run.py train                    # только обучение
  python run.py predict                 # только предсказание (нужна обученная модель)
  python run.py train predict           # обучение затем предсказание
"""
import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import RAW_DIR, MODELS_DIR
from src.train import run_train
from src.predict import run_predict


def main():
    parser = argparse.ArgumentParser(description="Sberbank Housing — пайплайн обучения и предсказания")
    parser.add_argument(
        "action",
        nargs="+",
        choices=["train", "predict"],
        help="train — обучить модель, predict — построить сабмит",
    )
    parser.add_argument("--data-dir", type=Path, default=RAW_DIR, help="Папка с train.csv, test.csv, macro.csv")
    parser.add_argument("--model-name", default="model", help="Имя файла модели (без расширения)")
    parser.add_argument("--submission-name", default="submission", help="Имя файла сабмита (без .csv)")
    parser.add_argument("--no-macro", action="store_true", help="Не подмешивать macro.csv")
    args = parser.parse_args()

    feat_kwargs = {
        "add_derived": True,
        "fill_na_numeric": "median",
        "categorical_strategy": "ordinal",
        "drop_high_missing": 0.9,
    }

    for act in args.action:
        if act == "train":
            print("Обучение модели...")
            model, X_train, X_test, feature_names, metrics = run_train(
                raw_dir=args.data_dir,
                with_macro=not args.no_macro,
                model_type="lgbm",
                use_time_split=True,
                save_model=True,
                model_name=args.model_name,
                **feat_kwargs,
            )
            print("Метрики:", metrics)
            print(f"Модель сохранена: {MODELS_DIR / f'{args.model_name}.joblib'}")
        elif act == "predict":
            print("Построение предсказаний...")
            sub = run_predict(
                model_name=args.model_name,
                raw_dir=args.data_dir,
                with_macro=not args.no_macro,
                submission_name=args.submission_name,
                **feat_kwargs,
            )
            print(f"Сабмит сохранён: submissions/{args.submission_name}.csv, строк: {len(sub)}")


if __name__ == "__main__":
    main()
