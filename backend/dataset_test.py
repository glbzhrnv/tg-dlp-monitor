import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from app.services.analyzer import LeakAnalyzer

# Инициализация анализатора
analyzer = LeakAnalyzer()

y_true = []
y_pred = []
y_scores = []  # для хранения вероятностей

categories = []

print("=== ОБРАБОТКА DATASET ===")

# Читаем CSV
with open("dataset.csv", newline='', encoding="utf-8") as csvfile:

    reader = csv.DictReader(csvfile, delimiter=";")

    print("Колонки CSV:", reader.fieldnames)

    for row_num, row in enumerate(reader, 1):
        text = row.get("text", "")
        label_raw = row.get("label", "0")
        category = row.get("category", "unknown")

        if not text:
            print(f"Строка {row_num}: пропущена (пустой текст)")
            continue

        try:
            label = int(label_raw.strip())
        except:
            print(f"Строка {row_num}: пропущена (некорректный label: '{label_raw}')")
            continue

        # Анализ текста
        leaks, rubert_score = analyzer.scan_message_for_leaks(text)
        sx_score = analyzer.calculate_sx(
            leaks,
            rubert_score,
            text=text
        )

        prediction = 1 if sx_score > 0.55 else 0

        y_true.append(label)
        y_pred.append(prediction)
        y_scores.append(sx_score)

        categories.append(category)

print(f"\n=== СТАТИСТИКА ОБРАБОТКИ ===")
print(f"Всего обработано: {len(y_true)} строк")
print(f"True: 0={y_true.count(0)}, 1={y_true.count(1)}")
print(f"Pred: 0={y_pred.count(0)}, 1={y_pred.count(1)}")

# ДИАГНОСТИКА ДЛЯ ROC
print(f"\n=== ДИАГНОСТИКА ДЛЯ ROC КРИВОЙ ===")
print(f"Уникальные значения y_true: {sorted(set(y_true))}")
print(f"Количество уникальных y_scores: {len(set(y_scores))}")
print(f"Диапазон y_scores: min={min(y_scores):.4f}, max={max(y_scores):.4f}")
print(f"Среднее y_scores: {np.mean(y_scores):.4f}")
print(f"Стандартное отклонение y_scores: {np.std(y_scores):.4f}")

# Показываем распределение y_scores по классам
scores_class0 = [score for score, true in zip(y_scores, y_true) if true == 0]
scores_class1 = [score for score, true in zip(y_scores, y_true) if true == 1]
print(f"\ny_scores для класса 0: min={min(scores_class0):.4f}, max={max(scores_class0):.4f}, среднее={np.mean(scores_class0):.4f}")
print(f"y_scores для класса 1: min={min(scores_class1):.4f}, max={max(scores_class1):.4f}, среднее={np.mean(scores_class1):.4f}")

# Проверка на вырожденность
if len(set(y_scores)) < 2:
    print("\n❌ ПРЕДУПРЕЖДЕНИЕ: все y_scores одинаковые - ROC-кривая будет вырожденной!")
elif len(set(y_scores)) < 10:
    print(f"\n⚠️ ВНИМАНИЕ: только {len(set(y_scores))} уникальных значений y_scores - ROC-кривая может быть негладкой")

print("\n=== МЕТРИКИ ===")

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# AUC вычисляем по сырым scores
try:
    auc = roc_auc_score(y_true, y_scores)
    print(f"AUC вычислена успешно: {auc:.4f}")
except Exception as e:
    auc = 0
    print(f"Ошибка при вычислении AUC: {e}")

print(f"\nPrecision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"AUC-ROC: {auc:.3f}")

# =========================================
# РИСУНОК 3.26 — Распределение типов утечек
# =========================================

print("\nСоздание рисунка распределения категорий...")

df = pd.DataFrame({
    "category": categories
})

category_counts = df["category"].value_counts()

plt.figure(figsize=(14, 7))

x_positions = range(len(category_counts))

plt.bar(
    x_positions,
    category_counts.values,
    width=0.6
)

plt.xticks(
    x_positions,
    category_counts.index,
    rotation=45,
    ha="right"
)

plt.title("Распределение сообщений по типам утечек")
plt.xlabel("Тип утечки")
plt.ylabel("Количество сообщений")
plt.tight_layout()

plt.savefig("figure_3_26_category_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("Рисунок 3.26 сохранён")

# =========================================
# РИСУНОК 3.27 — Матрица ошибок
# =========================================

print("\nСоздание матрицы ошибок...")

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Матрица ошибок модели обнаружения утечек")
plt.tight_layout()
plt.savefig("figure_3_27_confusion_matrix.png", dpi=300)
plt.close()

print("Рисунок 3.27 сохранён")

# =========================================
# РИСУНОК 3.28 — ROC-кривая (ИСПРАВЛЕНА)
# =========================================

print("\nСоздание ROC-кривой...")

# Получаем ROC кривую
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

print(f"Количество точек ROC-кривой: {len(fpr)}")
print(f"FPR (первые 5): {fpr[:5]}")
print(f"TPR (первые 5): {tpr[:5]}")
print(f"Thresholds (первые 5): {thresholds[:5]}")

# Создаем улучшенный график
plt.figure(figsize=(10, 8))

# Основная ROC кривая
plt.plot(fpr, tpr, linewidth=2, color='darkorange', 
         label=f'ROC кривая (AUC = {auc:.3f})')

# Диагональная линия (случайный классификатор)
plt.plot([0, 1], [0, 1], linestyle='--', color='navy', 
         linewidth=2, label='Случайный классификатор (AUC = 0.5)')

# Отмечаем рабочий порог 0.55
threshold_idx = min(range(len(thresholds)), 
                    key=lambda i: abs(thresholds[i] - 0.55))
plt.scatter(fpr[threshold_idx], tpr[threshold_idx], 
           color='red', s=150, zorder=5,
           label=f'Порог 0.55 (FPR={fpr[threshold_idx]:.3f}, TPR={tpr[threshold_idx]:.3f})')

# Добавляем сетку значений для наглядности
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.05])

plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC-кривая модели обнаружения утечек', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)

# Добавляем текстовую информацию
plt.text(0.6, 0.2, f'AUC = {auc:.3f}', 
         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.savefig("figure_3_28_roc_curve.png", dpi=300)
plt.close()

print("Рисунок 3.28 сохранён")

# Дополнительно сохраняем данные ROC-кривой в CSV для анализа
roc_data = pd.DataFrame({
    'FPR': fpr,
    'TPR': tpr,
    'Threshold': thresholds
})
roc_data.to_csv('roc_curve_data.csv', index=False)
print("Данные ROC-кривой сохранены в roc_curve_data.csv")

# =========================================
# РИСУНОК 3.29 — Сравнение метрик
# =========================================

print("\nСоздание графика метрик...")

metrics_names = ['Precision', 'Recall', 'F1-score', 'AUC-ROC']
metrics_values = [precision, recall, f1, auc]

plt.figure(figsize=(10, 6))

# Используем разные цвета для наглядности
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
bars = plt.bar(metrics_names, metrics_values, width=0.6, color=colors)

# Добавляем значения на столбцы
for bar, value in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Сравнение метрик качества модели', fontsize=14, fontweight='bold')
plt.ylabel('Значение метрики', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3, axis='y', linestyle='--')
plt.tight_layout()
plt.savefig("figure_3_29_metrics_comparison.png", dpi=300)
plt.close()

print("Рисунок 3.29 сохранён")

# =========================================
# ДОПОЛНИТЕЛЬНЫЙ ГРАФИК: Распределение scores
# =========================================

print("\nСоздание графика распределения scores...")

plt.figure(figsize=(12, 5))

# Гистограмма для класса 0 (без утечек)
plt.subplot(1, 2, 1)
plt.hist(scores_class0, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('SX Score')
plt.ylabel('Частота')
plt.title(f'Распределение scores для класса 0 (без утечек)\n(n={len(scores_class0)})')
plt.grid(True, alpha=0.3)
plt.axvline(x=0.55, color='red', linestyle='--', linewidth=2, label='Порог 0.55')
plt.legend()

# Гистограмма для класса 1 (с утечками)
plt.subplot(1, 2, 2)
plt.hist(scores_class1, bins=30, alpha=0.7, color='red', edgecolor='black')
plt.xlabel('SX Score')
plt.ylabel('Частота')
plt.title(f'Распределение scores для класса 1 (с утечками)\n(n={len(scores_class1)})')
plt.grid(True, alpha=0.3)
plt.axvline(x=0.55, color='red', linestyle='--', linewidth=2, label='Порог 0.55')
plt.legend()

plt.tight_layout()
plt.savefig("figure_3_30_scores_distribution.png", dpi=300)
plt.close()

print("Рисунок 3.30 сохранён (распределение scores)")

print("\n=== ГОТОВО ===")
print("\nСозданные файлы:")
print("1. figure_3_26_category_distribution.png - распределение категорий")
print("2. figure_3_27_confusion_matrix.png - матрица ошибок")
print("3. figure_3_28_roc_curve.png - ROC-кривая (ИСПРАВЛЕНА)")
print("4. figure_3_29_metrics_comparison.png - сравнение метрик")
print("5. figure_3_30_scores_distribution.png - распределение scores")
print("6. roc_curve_data.csv - данные ROC-кривой")