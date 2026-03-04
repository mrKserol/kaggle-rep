# House Prices — Advanced Regression Techniques

Решение задачи [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques): предсказание цены дома по признакам (регрессия).

## Структура проекта

```
house_prices/
├── README.md
├── data/
│   └── raw/              # Исходные данные (train.csv, test.csv, data_description.txt, sample_submission.csv)
├── notebooks/
│   └── 01_house_prices_solution_pipeline.ipynb   # EDA, препроцессинг, модели, submission
├── src/
│   └── features.py       # Feature engineering (функция + sklearn-трансформер)
└── submissions/
    └── submission_catboost.csv   # Готовый файл для загрузки на Kaggle
```

## Данные

Скачать датасет можно на [странице соревнования](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). Файлы `train.csv`, `test.csv`, `data_description.txt` и `sample_submission.csv` положите в `data/raw/`.

## Зависимости

- Python 3.8+
- pandas, numpy
- scikit-learn
- catboost
- matplotlib, seaborn (для визуализаций в ноутбуке)

Установка в виртуальное окружение:

```bash
pip install pandas numpy scikit-learn catboost matplotlib seaborn
```

## Подход

- **Препроцессинг:** заполнение пропусков (медиана по соседству для `LotFrontage`, мода/медиана для остальных), приведение категориальных колонок.
- **Feature engineering** (`src/features.py`): суммарная площадь, возраст дома и ремонта, число «эквивалентных» ванных, индикаторы (бассейн, гараж, камин, подвал, второй этаж), взаимодействие качества и состояния.
- **Baseline:** Ridge (линейная регрессия с L2-регуляризацией) в пайплайне с препроцессингом и OHE, оценка по 5-fold CV (RMSE).
- **Основная модель:** CatBoost с теми же фичами, ручной 5-fold CV и early stopping. Для сабмита используется модель, обученная на всём train.

## Результаты

Сравнение по среднему RMSE на кросс-валидации (5-fold):

| Модель   | RMSE (CV mean) |
|----------|----------------|
| Baseline (Ridge) | ~32 683 |
| CatBoost | ~25 998 |

Итоговый сабмит формируется предсказаниями CatBoost.
