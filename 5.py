import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

digits = load_digits()
X, y = digits.data, digits.target

models = {
    'Drzewo (bez limitu)': DecisionTreeClassifier(random_state=42),
    'Drzewo (max_depth=10)': DecisionTreeClassifier(random_state=42, max_depth=10),
    'RF (10 drzew)': RandomForestClassifier(n_estimators=10, random_state=42),
    'RF (50 drzew)': RandomForestClassifier(n_estimators=50, random_state=42),
    'RF (100 drzew)': RandomForestClassifier(n_estimators=100, random_state=42),
    'RF (200 drzew)': RandomForestClassifier(n_estimators=200, random_state=42)
}

results = {}
print("Obliczam cross-validation dla każdego modelu...")
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=10)
    results[name] = cv_scores
    print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


plt.figure(figsize=(12,6))
plt.boxplot(results.values(), tick_labels=results.keys())
plt.ylabel("Dokładność")
plt.title("Porównanie modeli - zbiór Digits (10-flod CV)")
plt.xticks()
plt.grid(True)
plt.tight_layout()
plt.show()