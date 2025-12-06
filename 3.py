import numpy as np
from sklearn.datasets import load_digits


digits = load_digits()
X, y = digits.data, digits.target

print("Zbiór Digits:")
print(f"Liczba próbek: {X.shape[0]}")
print(f"Liczba cech:   {X.shape[1]}")
print(f"Liczba klas:   {len(np.unique(y))}")

