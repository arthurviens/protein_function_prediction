"""
This is a boilerplate pipeline 'XGBoost'
generated using Kedro 0.17.6
"""
import logging
from typing import Dict, Any


from xgboost import XGBClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
)
import numpy as np


def train_model(X_train: np.array, y_train: np.array, parameters: Dict[str, Any]) -> XGBClassifier:
    """
    Node for training a SVC model given data provided to this function as the time of execution.
    """
    params2keep = ["colsample_bytree", "gamma", "max_depth", "min_child_weight", 
    "n_estimators", "subsample", "use_label_encoder", "objective", "eval_metric"]
    kept_params = {}
    for param in parameters:
        if param in params2keep:
            kept_params[param] = parameters[param]
    model = XGBClassifier(**kept_params)
    model.fit(X_train, y_train)
    return model

def predict(model: XGBClassifier, X_test: np.array) -> np.array:
    """
    Node for making predictions given a pre-trained model and a test data set.
    """
    return model.predict(X_test)


def report_scores(y_true: np.array, y_pred: np.array) -> None:
    """
    Node for reporting the scores of the predictions performed by previous node.
    """
    target = np.where(y_pred > 0.5, 1, 0)
    log = logging.getLogger(__name__)
    log.info("Model 'XGBoost' AUC : {}".format(roc_auc_score(y_true, y_pred)))
    log.info("Model 'XGBoost' accuracy : {}".format(accuracy_score(y_true, target)))
    log.info("Model 'XGBoost' precision : {}".format(precision_score(y_true, target)))
    log.info(
        "Model 'XGBoost' balanced accuracy : {}".format(balanced_accuracy_score(y_true, target))
    )
