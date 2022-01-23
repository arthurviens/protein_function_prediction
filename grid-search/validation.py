from traceback import print_exc

import numpy as np
import pandas as pandas
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV

from in_out import *
from preprocessing import *
from models import *

# Colors
class bcolors:
    HEADER = "\033[95m"  # Pink
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"  # Yellow
    FAIL = "\033[91m"  # Red
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


header = lambda x: bcolors.HEADER + x + bcolors.ENDC
blue = lambda x: bcolors.OKBLUE + x + bcolors.ENDC
cyan = lambda x: bcolors.OKCYAN + x + bcolors.ENDC
green = lambda x: bcolors.OKGREEN + x + bcolors.ENDC
warning = lambda x: bcolors.WARNING + x + bcolors.ENDC
fail = lambda x: bcolors.FAIL + x + bcolors.ENDC
bold = lambda x: bcolors.BOLD + x + bcolors.ENDC
underline = lambda x: bcolors.UNDERLINE + x + bcolors.ENDC

# For safety
HKEYS = tuple(HYPERPARAMETERS.keys())
MKEYS = tuple(MODELS.keys())
for key in KEYS:
    assert key in HKEYS, "{} not found in HYPERPARAMETERS (keys={})".format(key, HKEYS)
    assert key in MKEYS, "{} not found in MODELS (keys={})".format(key, MKEYS)

# ===========

# Load data
X_train, y_train = load_shrunk_data()
# X_train, y_train = load_data()[:2]

best_clf = {key: {} for key in SCALERS}

print(header("Start"))

for scaler_type in SCALERS:
    # Scale data
    print(underline("Current scaler : '{}'".format(scaler_type)))
    scaler = SCALERS[scaler_type]
    # scaler.fit(X_train)
    # X_train_scaled = scaler.transform(X_train)

    for key in KEYS:
        print(underline("Current model : '{}'".format(key)))
        print("Hyperparameters : {}".format(HYPERPARAMETERS[key]))
        model = Pipeline([("scaler", scaler), ("model", MODELS[key])])
        clf = GridSearchCV(
            model, HYPERPARAMETERS[key], cv=5, scoring="balanced_accuracy", n_jobs=-1
        )
        try:
            # Search best estimators
            clf.fit(X_train, y_train)
            best_clf[scaler_type][key] = clf
        except Exception:
            print_exc()

print("")
print(bold("Finished"))
print("")


# Print results
print(header("Results"))
for scaler_type in SCALERS:
    print(blue("With the scaler : '{}'".format(scaler_type)))
    for key in KEYS:
        clf = best_clf[scaler_type][key]
        try:
            print(underline("Estimator : {}".format(key)))
            print("Best estimators : {}".format(clf.best_estimator_))
            print(cyan("Best score : {}".format(clf.best_score_)))
            print("")
        except:
            print(fail("Classifier {} with scaler '{}' didn't work.".format(key, scaler_type)))


# Rank estimators
print("")
print(header("Best classifiers"))
print("")
try:
    list_clf = [
        (scaler_type, key, best_clf[scaler_type][key].best_score_)
        for scaler_type in SCALERS
        for key in KEYS
    ]
    sorted_list_clf = sorted(list_clf, key=lambda x: x[2], reverse=True)
    gradient = len(sorted_list_clf) // 3
    for i, (scaler_type, key, best_score) in enumerate(sorted_list_clf):
        if i <= gradient:
            color = green
        elif i <= 2 * gradient:
            color = warning
        else:
            color = fail
        print(color("Rank {} : {} {} {}".format(i, scaler_type, key, best_score)))
except Exception:
    print_exc()

print("End.")
