# train_rubert.py — надёжная версия с диагностикой и защитой
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import os

# Пути (поменяй, если нужно)
DATA_PATH = "dataset.csv"  # или полный путь: r"services\dataset.csv"
MODEL_NAME = "cointegrated/rubert-tiny2"
OUTPUT_DIR = "models/rubert-leak"
NUM_EPOCHS = 4
BATCH_SIZE = 16

# 1. Диагностика и чтение файла
print(f"Пытаемся прочитать: {DATA_PATH}")

try:
    # Пробуем разные варианты чтения
    df = pd.read_csv(
        DATA_PATH,
        sep=';',                    # основной разделитель в твоём файле
        quotechar='"',              # поля в двойных кавычках
        doublequote=True,           # двойные кавычки внутри текста
        escapechar='\\',            # экранирование
        encoding='utf-8',           # кодировка
        on_bad_lines='warn',        # предупреждать, но не падать
        engine='python'             # python-парсер лучше справляется с кавычками
    )
    print("\nУспешно прочитано строк:", len(df))
    print("Столбцы:", df.columns.tolist())
    print("\nПервые 5 строк:\n", df.head())
except Exception as e:
    print("Ошибка чтения файла:", e)
    print("Попробуем без заголовка...")
    try:
        df = pd.read_csv(DATA_PATH, sep=';', header=None, encoding='utf-8')
        df.columns = ['id', 'text', 'label', 'category', 'comment']
        print("Прочитано без заголовка, столбцы присвоены вручную")
    except Exception as e2:
        print("Всё ещё ошибка:", e2)
        exit(1)

# 2. Очистка и проверка
if 'text' not in df.columns or 'label' not in df.columns:
    print("Ошибка: нет столбцов 'text' или 'label'. Показываю столбцы:")
    print(df.columns)
    exit(1)

df = df[["text", "label"]].dropna()
print("После очистки строк:", len(df))

print("Типы столбцов до очистки:", df.dtypes)

# Очистка label: оставляем только 0 и 1, остальное отбрасываем
df['label'] = pd.to_numeric(df['label'], errors='coerce')  # некорректные → NaN
df = df[df['label'].isin([0, 1])]  # оставляем только 0 и 1
df['label'] = df['label'].astype(int)

print("После очистки label — строк:", len(df))
print("Уникальные значения в label:", df['label'].unique())

df = df[["text", "label"]].dropna()
print("После финальной очистки строк:", len(df))

# Проверка типов
df['label'] = df['label'].astype(int)  # на всякий случай приводим к int

# 3. Разделение
train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df["label"]
)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 4. Токенизация
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_tokenized = train_dataset.map(tokenize, batched=True)
val_tokenized = val_dataset.map(tokenize, batched=True)

train_tokenized = train_tokenized.rename_column("label", "labels")
val_tokenized = val_tokenized.rename_column("label", "labels")

train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 5. Модель
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    problem_type="single_label_classification"
)

# 6. Аргументы обучения
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(-1)
    return {"accuracy": accuracy_score(labels, predictions)}

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,  # ← ключевая строка
)

trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "best"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best"))

print(f"RuBERT обучен и сохранён в {os.path.join(OUTPUT_DIR, 'best')}")