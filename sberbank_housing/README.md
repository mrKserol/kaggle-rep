# Sberbank Russian Housing Market

Решение [Kaggle-соревнования](https://www.kaggle.com/competitions/sberbank-russian-housing-market): предсказание цены сделки с недвижимостью (`price_doc`) по данным Сбербанка (Москва и область, 2011–2016). Регрессия, метрика — RMSE.

---

## Структура репозитория

```
sberbank_housing/
├── README.md
├── requirements.txt
├── run.py                 # CLI: обучение и/или предсказание
├── data/
│   └── raw/               # train.csv, test.csv, macro.csv, sample_submission.csv
├── notebooks/
│   ├── eda.ipynb          # Разведка данных
│   └── train_model.ipynb  # Обучение модели и сабмит
├── src/                   # Пайплайн
│   ├── config.py          # Пути, константы, параметры моделей
│   ├── data.py            # Загрузка train/test/macro, джойн по дате
│   ├── features.py        # Фичи: дата, площади, кодирование, пропуски
│   ├── train.py           # Разбиение по времени, LightGBM/XGBoost, сохранение
│   └── predict.py         # Предсказание, формирование submission.csv
├── models/                 # Сохранённые модели (*.joblib)
└── submissions/            # Готовые сабмиты для Kaggle
```

---

## Данные

Файлы из Kaggle положите в `data/raw/`:

| Файл | Описание |
|------|----------|
| `train.csv` | Обучающая выборка: id, timestamp, признаки объекта и района, **price_doc** (целевая) |
| `test.csv` | Тестовая выборка (без price_doc) |
| `macro.csv` | Макроэкономика по датам (нефть, курс, ВВП, ипотека и т.д.) |
| `sample_submission.csv` | Шаблон: id, price_doc |

Пайплайн объединяет train/test с macro по полю `timestamp`. Описание признаков — в `data/raw/data_dictionary.txt`.

---

## Установка и запуск

**Зависимости:**

```bash
pip install -r requirements.txt
```

**Обучение и предсказание из терминала:**

```bash
python run.py train          # обучить модель → models/model.joblib
python run.py predict        # предсказать по test → submissions/submission.csv
python run.py train predict  # оба шага подряд
```

Опции: `--data-dir`, `--model-name`, `--submission-name`, `--no-macro`.

**Ноутбуки:**

- `notebooks/eda.ipynb` — загрузка через `src`, разведка целевой, пропусков, районов, корреляций, подготовка фичей.
- `notebooks/train_model.ipynb` — те же данные и фичи, разбиение по времени, обучение LightGBM, метрики, важность признаков, сохранение модели и сабмита.

Запускать из корня проекта или из папки `notebooks/` (путь к `src` в ноутбуках подставляется автоматически).

---

## Пайплайн в двух словах

1. **Загрузка:** train + test + macro, джойн по дате сделки.
2. **Фичи:** признаки из даты (год, месяц, квартал), доли площадей (life_sq/full_sq и др.), категории в ordinal; пропуски — медиана; колонки с более 90% пропусков отбрасываются.
3. **Цель:** обучаем по `log1p(price_doc)`, предсказания переводим в рубли через `expm1`.
4. **Валидация:** разбиение по времени (последние 20% — валидация).
5. **Модель:** LightGBM (по умолчанию), при необходимости XGBoost; модель и список признаков сохраняются в `models/`, сабмит — в `submissions/`.

---

## Ссылки

- [Соревнование на Kaggle](https://www.kaggle.com/competitions/sberbank-russian-housing-market)
- [Описание признаков](data/raw/data_dictionary.txt) в репозитории
