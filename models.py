from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


KEYS = ("svd", "logit", "randomforest", "decisiontree")

HYPERPARAMETERS = {
    "svc": {
        "C": [0.001, 0.01, 0.1, 1, 10, 20],
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "gamma": ["scale", "auto"],
    },
    "logit": {
        "penality": ["l1", "l2", "elasticnet", "none"],
        "C": [0.001, 0.01, 0.1, 1, 10, 20],
    },
    "randomforest": {
        "n_estimators": [5, 10, 20, 50, 100],
        "criterion": ["gini", "entropy"],
        "max_features": {"auto", "sqrt", "log2"},
    },
    "decisiontree": {
        "criterion": ["gini", "entropy"],
        "max_features": ["auto", "sqrt", "log2"],
    },
}

MODELS = {
    "svc": SVC(random_state=42, max_iter=10000),
    "logit": LogisticRegression(max_iter=5000),
    "randomforest": RandomForestClassifier(random_state=42),
    "decisiontree": DecisionTreeClassifier(random_state=42),
}
