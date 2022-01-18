from traceback import print_exc

import numpy as np
import pandas as pandas
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV

from in_out import *
from preprocessing import *
from models import *

# For safety
HKEYS = tuple(HYPERPARAMETERS.keys())
MKEYS = tuple(MODELS.keys())
for key in KEYS:
    assert key in HKEYS, "{} not found in HYPERPARAMETERS (keys={})".format(key, HKEYS)
    assert key in MKEYS, "{} not found in MODELS (keys={})".format(key, MKEYS)

# ===========

# Load data
X_train, y_train = load_shrunk_data()

best_clf = {key: {} for key in SCALERS}

print("Start :")

for scaler_type in SCALERS:
    # Scale data
    print("Current scaler : '{}'".format(scaler_type))
    scaler = SCALERS[scaler_type]
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    for key in KEYS:
        print("Current model : '{}'".format(key))
        print("Hyperparameters : {}".format(HYPERPARAMETERS[key]))
        clf = GridSearchCV(
            MODELS[key], HYPERPARAMETERS[key], cv=5, scoring="balanced_accuracy", n_jobs=-1
        )
        try:
            # Search best estimators
            clf.fit(X_train_scaled, y_train)
            best_clf[scaler_type][key] = clf
        except Exception:
            print_exc()

print("\nFinished.\n")


# Print results
print("Results :")
for scaler_type in SCALERS:
    print("With the scaler : '{}'".format(scaler_type))
    for key in KEYS:
        clf = best_clf[scaler_type][key]
        try:
            print("Best estimators : {}".format(clf.best_estimator_))
            print("Best score : ".format(clf.best_score_))
        except:
            print("Classifier {} with scaler '{}' didn't work.".format(key, scaler_type))


# Rank estimators
print("\nBest classifiers\n")
try:
    list_clf = [
        (scaler_type, key, best_clf[scaler_type][key].best_score_)
        for scaler_type in SCALERS
        for key in KEYS
    ]
    sorted_list_clf = sorted(list_clf, key=lambda x: x[2], reverse=True)
    for i, (scaler_type, key, best_score) in enumerate(sorted_list_clf):
        print("Rank", i, ":", scaler_type, key, best_score)
except Exception:
    print_exc()

print("End.")
