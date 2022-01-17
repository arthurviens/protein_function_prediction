from traceback import print_exc

import numpy as np
import pandas as pandas
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV

from in_out import *
from preprocessing import *
from models import *

# Load data
X_train, y_train = load_shrunk_data()
# X_train, y_train, X_test, y_test = get_train_test(X, y)

best_clf = {key: {} for key in SCALERS}

print("Start :")

for scaler_type in SCALERS:
    # Scale data
    print("Current scaler : '{}'".format(scaler_type))
    scaler = SCALERS[scaler_type]
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    for key in KEYS:
        print("Current model : '{}'".format())
        print("Hyperparameters : {}".format())
        clf = GridSearchCV(
            MODELS[key], HYPERPARAMETERS[key], cv=5, scoring="balanced_accuracy", n_jobs=-1
        )
        try:
            # Search best estimators
            clf.fit(X_train_scaled, y_train)
        except Exception:
            print_exc()
        best_clf[scaler_type][key] = clf

print("\nFinished.\n")

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

print("\nBest classifiers\n")
try:
    list_clf = [
        (scaler_type, key, best_clf[scaler_type][key])
        for scaler_type in SCALERS
        for key in KEYS
    ]
    sorted_list_clf = sorted(list_clf, key=lambda x: x[2].best_score_)
    for i, (scaler_type, key, _) in sorted_list_clf:
        print("{} {} {}".format(i, scaler_type, key))
except Exception:
    print_exc()

print("End.")
