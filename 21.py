import lightgbm as lgb
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 1. WCZYTANIE DANYCH
iris = load_iris()
X, y = pd.DataFrame(iris.data, columns=iris.feature_names), iris.target

# 2. PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. UTWORZENIE I TRENOWANIE MODELU LightGBM
lgb_classifier = lgb.LGBMClassifier(
    n_estimators=100,        # liczba drzew
    num_leaves=31,           # max liczba liści w drzewie (kluczowy parametr!)
    max_depth=-1,            # -1 = bez limitu (kontrolowane przez num_leaves)
    learning_rate=0.1,       # tempo uczenia
    random_state=42,
    n_jobs=-1,
    verbose=-1               # wyłącz komunikaty
)

lgb_classifier.fit(X_train, y_train)

# 4. PREDYKCJA I EWALUACJA
y_pred = lgb_classifier.predict(X_test)

print("LightGBM - ZBIÓR IRIS")
print("=" * 50)
print(f"Dokładność treningowa: {lgb_classifier.score(X_train, y_train):.4f}")
print(f"Dokładność testowa:    {accuracy_score(y_test, y_pred):.4f}")

# Cross-validation
cv_scores = cross_val_score(lgb_classifier, X, y, cv=10)
print(f"\nCross-validation (10-fold):")
print(f"Średnia: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Raport klasyfikacji
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
