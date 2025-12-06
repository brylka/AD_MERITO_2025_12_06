from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
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