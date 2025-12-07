import xgboost as xgb
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
X, y = digits.data, digits.target

print("ZBIÓR DIGITS")
print("-" * 40)
print(f"Liczba próbek: {X.shape[0]}")
print(f"Liczba cech:   {X.shape[1]}")
print(f"Liczba klas:   {len(np.unique(y))}")

# 2. PODZIAŁ DANYCH
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. MODELE DO PORÓWNANIA
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest (100)': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost (100)': xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    'XGBoost (200, lr=0.05)': xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
}

# 4. PORÓWNANIE
print("\n" + "=" * 80)
print("PORÓWNANIE MODELI - DIGITS")
print("=" * 80)
print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12} {'Czas [s]':<10}")
print("-" * 80)

results = {}
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=5)

    results[name] = cv_scores
    print(f"{name:<25} {train_acc:<12.4f} {test_acc:<12.4f} {cv_scores.mean():<12.4f} {train_time:<10.4f}")

# # 5. WIZUALIZACJA BOXPLOT
# plt.figure(figsize=(12, 6))
# plt.boxplot(results.values(), tick_labels=results.keys())
# plt.ylabel('Dokładność')
# plt.title('Porównanie modeli - zbiór digits (5-fold CV)')
# plt.xticks(rotation=15, ha='right')
# plt.grid(True, axis='y')
# plt.tight_layout()
# plt.show()

# # 6. SZCZEGÓŁOWA ANALIZA NAJLEPSZEGO XGBoost
# best_xgb = models['XGBoost (200, lr=0.05)']
# y_pred = best_xgb.predict(X_test)
#
# print("\n" + "=" * 50)
# print("RAPORT KLASYFIKACJI - XGBoost (200, lr=0.05)")
# print("=" * 50)
# print(classification_report(y_test, y_pred))

# # 7. MACIERZ POMYŁEK
# plt.figure(figsize=(10, 8))
# cm = confusion_matrix(y_test, y_pred)
# plt.imshow(cm, interpolation='nearest', cmap='Blues')
# plt.title('Macierz pomyłek - XGBoost (digits)')
# plt.colorbar()
# plt.xlabel('Przewidziana klasa')
# plt.ylabel('Prawdziwa klasa')
#
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         plt.text(j, i, cm[i, j], ha='center', va='center',
#                  color='white' if cm[i, j] > cm.max() / 2 else 'black')
#
# plt.xticks(range(10))
# plt.yticks(range(10))
# plt.tight_layout()
# plt.show()
