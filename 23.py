import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time

# 1. WCZYTANIE DANYCH
digits = load_digits()
X, y = pd.DataFrame(digits.data, columns=digits.feature_names), digits.target

print("ZBIÓR DIGITS")
print("-" * 40)
print(f"Liczba próbek: {X.shape[0]}")
print(f"Liczba cech:   {X.shape[1]}")
print(f"Liczba klas:   {len(np.unique(y))}")

# 2. PODZIAŁ DANYCH
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. DEFINICJA MODELI
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest (100)': RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'Random Forest (200)': RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    'XGBoost (100)': xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=6,
        random_state=42, n_jobs=-1, verbosity=0
    ),
    'XGBoost (200)': xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=42, n_jobs=-1, verbosity=0
    ),
    'LightGBM (100)': lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.1, num_leaves=31,
        random_state=42, n_jobs=-1, verbose=-1
    ),
    'LightGBM (200)': lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=50,
        random_state=42, n_jobs=-1, verbose=-1
    )
}

# 4. TRENOWANIE I EWALUACJA
print("\n" + "=" * 90)
print("PORÓWNANIE MODELI - DIGITS")
print("=" * 90)
print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<10} {'Czas [s]':<10}")
print("-" * 90)

results = {}
times = {}
trained_models = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=5)

    results[name] = cv_scores
    times[name] = train_time
    trained_models[name] = model

    print(
        f"{name:<25} {train_acc:<12.4f} {test_acc:<12.4f} {cv_scores.mean():<12.4f} {cv_scores.std():<10.4f} {train_time:<10.4f}")

# # 5. WIZUALIZACJA - BOXPLOT DOKŁADNOŚCI
# plt.figure(figsize=(14, 6))
# plt.boxplot(results.values(), tick_labels=results.keys())
# plt.ylabel('Dokładność')
# plt.title('Porównanie modeli ensemble - zbiór digits (5-fold CV)')
# plt.xticks(rotation=30, ha='right')
# plt.grid(True, axis='y')
# plt.tight_layout()
# plt.show()
#
# # 6. WIZUALIZACJA - CZAS TRENINGU
# plt.figure(figsize=(12, 5))
# colors = ['gray', 'forestgreen', 'forestgreen', 'steelblue', 'steelblue', 'darkorange', 'darkorange']
# plt.bar(times.keys(), times.values(), color=colors)
# plt.ylabel('Czas treningu [s]')
# plt.title('Czas treningu modeli')
# plt.xticks(rotation=30, ha='right')
# plt.tight_layout()
# plt.show()
#
# # 7. PODSUMOWANIE
# print("\n" + "=" * 50)
# print("RANKING WG ŚREDNIEJ DOKŁADNOŚCI CV")
# print("=" * 50)
# ranking = sorted(results.items(), key=lambda x: x[1].mean(), reverse=True)
# for i, (name, scores) in enumerate(ranking, 1):
#     print(f"{i}. {name:<25} {scores.mean():.4f}")
