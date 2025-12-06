from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

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
