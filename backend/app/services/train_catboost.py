# train_catboost.py — обучение CatBoost с защитой от битых строк
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import joblib
import os
from rubert_inference import get_rubert_score  # твой файл с RuBERT

# Пути
DATA_PATH = "dataset.csv"
RUBERT_MODEL_PATH = "C:/Users/minez/tg-dlp-monitor/backend/models/rubert-leak/best"
OUTPUT_MODEL_PATH = "C:/Users/minez/tg-dlp-monitor/backend/models/catboost_leak.pkl"

print(f"Пытаемся прочитать датасет: {DATA_PATH}")

# Чтение с максимальной защитой
df = pd.read_csv(
    DATA_PATH,
    sep=';',                  # разделитель
    quotechar='"',
    doublequote=True,
    escapechar='\\',
    encoding='utf-8',
    on_bad_lines='warn',      # предупреждать о битых строках
    engine='python'
)

print("Прочитано строк:", len(df))
print("Столбцы:", df.columns.tolist())
print("\nПервые 5 строк:\n", df.head())

# Очистка label — приводим к числу, битые → NaN → удаляем
df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label', 'text'])
df['label'] = df['label'].astype(int)

print("После очистки строк:", len(df))
print("Уникальные значения label:", df['label'].unique())

# Добавляем rubert_score (это займёт время, но нужно для фич)
print("Считаем rubert_score...")
df['rubert_score'] = df['text'].apply(get_rubert_score)

# Фичи
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

# 4. Разделение
X = df[['rubert_score', 'text_length', 'word_count']]  # можно добавить больше фич позже
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 5. Обучение CatBoost
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100
)

train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

model.fit(
    train_pool,
    eval_set=val_pool,
    early_stopping_rounds=50,
    use_best_model=True
)

# 6. Оценка
train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Val accuracy: {val_acc:.4f}")

# 7. Сохранение
joblib.dump(model, OUTPUT_MODEL_PATH)
print(f"CatBoost сохранён: {OUTPUT_MODEL_PATH}")