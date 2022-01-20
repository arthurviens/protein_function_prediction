import logging
from typing import Dict, Any

from sklearn.svm import SVC
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
)
import numpy as np


def train_model(X_train: np.array, y_train: np.array, parameters: Dict[str, Any]) -> SVC:
    """
    Node for training a SVC model given data provided to this function as the time of execution.
    """
    params2remove = ("n_components_pca", "scaler_type", "test_data_ratio")
    for param in params2remove:
        if param in parameters:
            parameters.pop(param)
    model = SVC(**parameters)
    model.fit(X_train, y_train)
    return model


def predict(model: SVC, X_test: np.array) -> np.array:
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
    log.info("Model 'SVC' AUC : {}".format(roc_auc_score(y_true, y_pred)))
    log.info("Model 'SVC' accuracy : {}".format(accuracy_score(y_true, target)))
    log.info("Model 'SVC' precision : {}".format(precision_score(y_true, target)))
    log.info(
        "Model 'SVC' balanced accuracy : {}".format(balanced_accuracy_score(y_true, target))
    )
