"""
train_model.py
--------------
Model training utilities for the NIDS-XAI project.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def get_model(model_name: str = "random_forest"):
    """Return a model instance by name."""
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name]


def train(model, X_train, y_train):
    """Fit a model and return it."""
    model.fit(X_train, y_train)
    return model
