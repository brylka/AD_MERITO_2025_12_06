import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_cv = cross_val_score(rf, X, y, cv=10)

print("Ranfom Forest (100 drzew):")
print(f"Dokładność treningowa: {rf.score(X_train, y_train):.4f}")
print(f"Dokładność testowa:    {rf.score(X_test, y_test):.4f}")
print(f"CV średnia:            {rf_cv.mean():.4f} (+/- {tree_cv.std():.4f})")
print(f"OOB score:             {rf.oob_score_:.4f}")

y_pred = rf.predict(X_test)
print(f"Raport klasyfikacji (Random Forest):")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10,8))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Macierz pomyłek - Random Forest - zbiór Digits")
plt.xlabel("Przewidziana cyfra")
plt.ylabel("Prawdziwa cyfra")
plt.colorbar()
plt.xticks(range(10))
plt.yticks(range(10))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j,i,cm[i,j], ha="center", va="center",
                 color="white" if cm[i,j] > cm.max()/2 else "black")

plt.tight_layout()
plt.show()