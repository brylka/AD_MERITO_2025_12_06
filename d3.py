import joblib
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

# wczytanie danych - digits
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

X = X / 255

# tworzenie i trening modelu
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# zapisanie modelu do pliku
joblib.dump(model, 'model_mnist.joblib')
print("Zapisano model do pliku!")