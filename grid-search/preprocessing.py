import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def get_train_test(X, y):
    full = np.concatenate([X, y.reshape(y.shape[0], 1)], axis=1)
    train_set, test_set = train_test_split(full, test_size=0.2, random_state=42)
    X_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    X_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    return X_train, y_train, X_test, y_test


SCALERS = {
    "standard": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=0.95))]),
    "minmax": Pipeline([("scaler", MinMaxScaler()), ("pca", PCA(n_components=0.95))]),
}
