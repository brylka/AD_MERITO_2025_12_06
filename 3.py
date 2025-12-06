import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_cv = cross_val_score(tree, X, y, cv=10)

print("Pojedyncze drzewo:")
print(f"Dokładność treningowa: {tree.score(X_train, y_train):.4f}")
print(f"Dokładność testowa:    {tree.score(X_test, y_test):.4f}")
print(f"CV średnia:            {tree_cv.mean():.4f} (+/- {tree_cv.std():.4f})")

