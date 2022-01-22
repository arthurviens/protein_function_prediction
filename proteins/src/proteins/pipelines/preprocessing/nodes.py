from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

import numpy as np


def scale_data(
    X_train: np.array,  # train data set
    X_test: np.array,  # test data set
    X_test_competition: np.array,  # data for competition
    X_valid: np.array,  # data for competition
    scaler: str,  # only "minmax" or "standard" are implemented
    n_components: float,  # between 0 and 1
):
    """
    Scale data and apply a PCA on it
    """
    if scaler == "minmax":
        scaler = MinMaxScaler()
    elif scaler == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("'{}' is not implemented (see preprocessing/nodes.py)".format(scaler))

    preprocessing_pipeline = Pipeline(
        [("scaler", scaler), ("pca", PCA(n_components=n_components))]
    )
    preprocessing_pipeline.fit(X_train)
    X_train = preprocessing_pipeline.transform(X_train)
    X_test = preprocessing_pipeline.transform(X_test)
    X_test_competition = preprocessing_pipeline.transform(X_test_competition)
    X_valid = preprocessing_pipeline.transform(X_valid)
    return X_train, X_test, X_test_competition, X_valid
