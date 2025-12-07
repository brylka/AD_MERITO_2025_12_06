import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
import time
import numpy as np

# Wczytanie danych
iris = load_iris()
X, y = pd.DataFrame(iris.data, columns=iris.feature_names), iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Modele do porównania
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                 random_state=42, n_jobs=-1, verbosity=0),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                                   random_state=42, n_jobs=-1, verbose=-1)
}

print("PORÓWNANIE MODELI ENSEMBLE - IRIS")
print("=" * 80)
print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<10} {'Czas [s]':<10}")
print("-" * 80)

results = {}
times = {}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=10)

    results[name] = cv_scores
    times[name] = train_time

    print(
        f"{name:<20} {train_acc:<12.4f} {test_acc:<12.4f} {cv_scores.mean():<12.4f} {cv_scores.std():<10.4f} {train_time:<10.4f}")
