from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt


iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
random_forest.fit(X_train, y_train)

tree_train = single_tree.score(X_train, y_train)
tree_test = single_tree.score(X_test, y_test)
tree_cv = cross_val_score(single_tree, X, y, cv=10)

rf_train = random_forest.score(X_train, y_train)
rf_test = random_forest.score(X_test, y_test)
rf_cv = cross_val_score(random_forest, X, y, cv=10)

print("Pojedyncze drzewo:")
print(f"Dokładnośc treningowa: {tree_train:.4f}")
print(f"Dokładność testowa:    {tree_test:.4f}")
print(f"CV średnia:            {tree_cv.mean():.4f} (+/- {tree_cv.std():.4f})")

print("Random forest (100 drzew):")
print(f"Dokładnośc treningowa: {rf_train:.4f}")
print(f"Dokładność testowa:    {rf_test:.4f}")
print(f"CV średnia:            {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

print(f"Analiza różnic:")
print(f"Różnica overfitting:")
print(f"Drzewo: {(tree_train - tree_test):.4f}")
print(f"Las:    {(rf_train - rf_test):.4f}")
print(f"Stabilność (std):")
print(f"Drzewo: {tree_cv.std():.4f}")
print(f"Las:    {rf_cv.std():.4f}")

feature_importance = pd.DataFrame({
    'cecha': iris.feature_names,
    'ważność': random_forest.feature_importances_
}).sort_values('ważność', ascending=False)

print(f"Ważność cech:")
print(feature_importance.to_string(index=False))

plt.figure(figsize=(10,6))
plt.barh(feature_importance['cecha'], feature_importance['ważność'], color='green')
plt.xlabel('Ważność')
plt.title('Random Forest - Ważność cech (Irysy)')
plt.gca().invert_yaxis()
plt.show()

n_trees_range = [1, 5, 10, 25, 50, 100, 200, 300, 500]
train_scores = []
test_scores = []
oob_scores = []

for n_trees in n_trees_range:
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
    oob_scores.append(rf.oob_score_)

