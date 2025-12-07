import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 1. WCZYTANIE DANYCH
iris = load_iris()
X, y = iris.data, iris.target

# 2. PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. UTWORZENIE I TRENOWANIE MODELU XGBoost
xgb_classifier = xgb.XGBClassifier(
    n_estimators=100,        # liczba drzew (rund boostingu)
    max_depth=3,             # głębokość drzew (płytkie!)
    learning_rate=0.1,       # eta - tempo uczenia
    objective='multi:softmax',  # funkcja celu dla wieloklasowej klasyfikacji
    num_class=3,             # liczba klas
    random_state=42,
    n_jobs=-1,               # użyj wszystkich rdzeni
    verbosity=0              # wyłącz komunikaty
)

xgb_classifier.fit(X_train, y_train)

# 4. PREDYKCJA I EWALUACJA
y_pred = xgb_classifier.predict(X_test)

print("XGBoost - ZBIÓR IRIS")
print("=" * 50)
print(f"Dokładność treningowa: {xgb_classifier.score(X_train, y_train):.4f}")
print(f"Dokładność testowa:    {accuracy_score(y_test, y_pred):.4f}")

# Cross-validation
cv_scores = cross_val_score(xgb_classifier, X, y, cv=10)
print(f"\nCross-validation (10-fold):")
print(f"Średnia: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Raport klasyfikacji
print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
