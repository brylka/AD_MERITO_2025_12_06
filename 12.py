import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import time

# Wczytanie danych
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modele do porównania
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                 random_state=42, n_jobs=-1, verbosity=0)
}

print("PORÓWNANIE MODELI - IRIS")
print("=" * 70)
print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<10} {'Czas':<8}")
print("-" * 70)

for name, model in models.items():
    # Pomiar czasu
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Metryki
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=10)

    print(
        f"{name:<20} {train_acc:<12.4f} {test_acc:<12.4f} {cv_scores.mean():<12.4f} {cv_scores.std():<10.4f} {train_time:<8.4f}")
