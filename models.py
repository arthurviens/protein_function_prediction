from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


KEYS = ("svc", "logit_l2", "logit_l1", "logit_elasticnet", "randomforest", "decisiontree")

HYPERPARAMETERS = {
    "svc": {
        "C": [0.001, 0.01, 0.1, 1, 10, 20],
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "gamma": ["scale"],  # , "auto"],
    },
    "logit_l2": {
        "penalty": ["l2"],
        "C": [0.001, 0.01, 0.1, 1, 10, 20],
        "solver": ["lbfgs"],
    },
    "logit_l1": {
        "penalty": ["l1"],
        "C": [0.001, 0.01, 0.1, 1, 10, 20],
        "solver": ["saga"],
    },
    "logit_elasticnet": {
        "penalty": ["elasticnet"],
        "C": [0.001, 0.01, 0.1, 1, 10, 20],
        "l1_ratio": [0.3, 0.5, 0.8],
        "solver": ["saga"],
    },
    "randomforest": {
        "n_estimators": [5, 10, 20, 50, 100],
        "criterion": ["gini", "entropy"],
        "max_features": ["auto", "sqrt", "log2"],
    },
    "decisiontree": {
        "criterion": ["gini", "entropy"],
        "max_features": ["auto", "sqrt", "log2"],
    },
}

MODELS = {
    "svc": SVC(random_state=42, max_iter=10000),
    "logit_l2": LogisticRegression(max_iter=5000),
    "logit_l1": LogisticRegression(max_iter=5000),
    "logit_elasticnet": LogisticRegression(max_iter=5000),
    "randomforest": RandomForestClassifier(random_state=42),
    "decisiontree": DecisionTreeClassifier(random_state=42),
}
