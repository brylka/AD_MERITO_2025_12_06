import joblib
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

# wczytanie danych - digits
digits = load_digits()
X, y = digits.data, digits.target

# tworzenie i trening modelu
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# zapisanie modelu do pliku
joblib.dump(model, 'model.joblib')
print("Zapisano model do pliku!")