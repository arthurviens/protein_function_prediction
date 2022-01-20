from typing import List

import numpy as np
from sklearn.model_selection import train_test_split


def split_data(X: np.array, y: np.array, test_ratio: float) -> List[np.array]:
    """
    Node for splitting data set into training and test sets,
    each split into features and labels
    """
    full = np.concatenate([X, y.reshape(y.shape[0], 1)], axis=1)
    train_set, test_set = train_test_split(full, test_size=test_ratio, random_state=42)
    X_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    X_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    return X_train, y_train, X_test, y_test
