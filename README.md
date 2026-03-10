# Data Science Practice

В репозитории собраны задачи по Data Science, выполненные в рамках курсов Kaggle.  
Цель — демонстрация понимания базовых методов и работы с кодом.

---

## Задачи

| № | Задача | Каталог | Описание | Решение |
|---|--------|---------|----------|---------|
| 1 | **House Prices** | [house_prices/](house_prices/) | Предсказание цены дома по признакам (регрессия). [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) | ✅ Код, модели, сабмиты |
| 2 | **Sberbank Russian Housing** | [sberbank_housing/](sberbank_housing/) | Предсказание цены сделки с недвижимостью по данным Сбербанка (регрессия, RMSE). [Kaggle](https://www.kaggle.com/competitions/sberbank-russian-housing-market) | ✅ Код, модели, сабмиты |
| 3 | **Store Sales — Time Series** | [store_sales_forecasting/](store_sales_forecasting/) | Прогнозирование продаж магазинов (временные ряды). [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) | Структура проекта |

В каталогах **house_prices** и **sberbank_housing** лежат полные решения: ноутбуки, исходный код, обученные модели и готовые файлы для отправки на Kaggle. Подробное описание каждой задачи, структура проекта, зависимости и инструкции по запуску — в файле **README.md** внутри соответствующего каталога.

---

## Структура репозитория

```
kaggle_rep/
├── README.md                 # этот файл
├── house_prices/             # House Prices — Advanced Regression Techniques
│   └── README.md             # описание, подход, результаты
├── sberbank_housing/         # Sberbank Russian Housing Market
│   └── README.md             # описание, пайплайн, запуск
└── store_sales_forecasting/  # Store Sales — Time Series Forecasting
    └── README.md
```

---

## Технологии

- **Язык:** Python 3.8+
- **Библиотеки:** pandas, numpy, scikit-learn, CatBoost (house_prices); LightGBM / XGBoost (sberbank_housing); визуализация: matplotlib, seaborn.

Точный список зависимостей по каждой задаче см. в README подкаталога (в т.ч. `requirements.txt` в **sberbank_housing**).
