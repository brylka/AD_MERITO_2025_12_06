import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


digits = load_digits()
X, y = digits.data, digits.target

print("Zbiór Digits:")
print(f"Liczba próbek: {X.shape[0]}")
print(f"Liczba cech:   {X.shape[1]}")
print(f"Liczba klas:   {len(np.unique(y))}")

fig, axes = plt.subplots(2,5, figsize=(12,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='gray')
    ax.set_title(f"Cyfra: {digits.target[i]}")
    ax.axis('off')
plt.suptitle('Przykładowe cyfry ze zbioru Digits', fontsize=14)
plt.tight_layout()
plt.show()
