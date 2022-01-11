import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler, PolynomialFeatures
from sklearn.svm import *
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import *
from script import *


def load_shrunk_data(path="data/"):
    X = np.load(path + "shrink_0.npy")
    y = np.load(path + "shrink_1.npy")

    return X, y


def plot_corr_scatter(dfX):
    corr = dfX.corr()
    best = []
    for i in range(dfX.shape[1]):
        x = corr[i].sort_values(ascending=False)[1:2]
        r = x.iloc[0]
        index = x.index[0]
        if abs(r) >= 0.9:
            if index == i:
                index = corr[i].sort_values(ascending=False)[0:1].index[0]
            best.append((i, index, r))

    best = sorted(best, key=lambda e: e[2], reverse=True)
    print(best)
    best = best[::2]
    indices = {x for element in best[:5] for x in element[:2]}
    scatter_matrix(dfX[indices], figsize=(16, 16))
    plt.show()


def get_PCA_opt(X_scaled):
    pca = PCA()
    pca.fit(X_scaled)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    tol = 0.95
    d = np.argmax(cumsum >= tol) + 1
    plt.figure(figsize=(6, 4))
    plt.plot(cumsum, linewidth=3)
    # plt.axis([0, 952, 0, 1])
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    plt.plot([d, d], [0, tol], "k:")
    plt.plot([0, d], [tol, tol], "k:")
    plt.plot(d, tol, "ko")
    plt.grid(True)
    plt.show()


def reduce(X):
    X_scaled = pd.DataFrame(scale(X))
    pca_opt = PCA(n_components=0.95)
    return pca_opt.fit_transform(X_scaled)


def get_train_test(X, y):
    df = lambda x: pd.DataFrame(x)
    full = np.concatenate([X, y.reshape(y.shape[0], 1)], axis=1)
    train_set, test_set = train_test_split(full, test_size=0.2, random_state=42)
    X_train = train_set[:, :-1]
    y_train = train_set[:, -1]
    X_test = test_set[:, :-1]
    y_test = test_set[:, -1]
    return X_train, y_train, X_test, y_test


def model_1(X, y):
    X_train, y_train, X_test, y_test = get_train_test(X, y)
    svm_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            # ("poly_features", PolynomialFeatures(degree=2)),
            ("linear_svc", LinearSVC(C=0.01, loss="hinge", max_iter=5000)),
        ]
    )
    y_pred = cross_val_predict(svm_clf, X_train, y_train, cv=3, method="decision_function")
    print("Cross Validation")
    print("AUC :")
    print(roc_auc_score(y_train, y_pred))
    print("Accuracy :")
    print(accuracy_score(y_train, np.where(y_pred > 0.5, 1, 0)))
    print("Precision :")
    print(precision_score(y_train, np.where(y_pred > 0.5, 1, 0)))
    svm_clf.fit(X_train, y_train)
    y_tpred = svm_clf.predict(X_test)
    print("Model 80% / 20%")
    print("AUC :")
    print(roc_auc_score(y_test, y_tpred))
    print("Accuracy :")
    print(accuracy_score(y_test, np.where(y_tpred > 0.5, 1, 0)))
    print("Precision :")
    print(precision_score(y_test, np.where(y_tpred > 0.5, 1, 0)))


if __name__ == "__main__":
    # X, y = load_data()[:2]
    X, y = load_shrunk_data()
    X_train, y_train, X_test, y_test = get_train_test(X, y)
    # X = X[:, :50]

    svm_clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            (
                "svc",
                SVC(kernel="rbf", max_iter=5000),
            ),
        ]
    )

    y_pred = cross_val_predict(svm_clf, X_train, y_train, cv=3, method="decision_function")
    # y_pred = svm_clf.predict(X_test)
    print("Cross Validation")
    print("AUC :")
    print(roc_auc_score(y_train, y_pred))
    print("Accuracy :")
    print(accuracy_score(y_train, np.where(y_pred > 0.5, 1, 0)))
    print("Precision :")
    print(precision_score(y_train, np.where(y_pred > 0.5, 1, 0)))

    svm_clf.fit(X_train, y_train)
    y_tpred = svm_clf.predict(X_test)
    print("Model 80% / 20%")
    print("AUC :")
    print(roc_auc_score(y_test, y_tpred))
    print("Accuracy :")
    print(accuracy_score(y_test, np.where(y_tpred > 0.5, 1, 0)))
    print("Precision :")
    print(precision_score(y_test, np.where(y_tpred > 0.5, 1, 0)))

    # print()

    # print(scale(dfX).shape)

    # print(X_reduced.shape)

    # X_reduced = df(scaler.transform(df(X)))

    # X_reduced.describe().loc["max"].hist()
    # plt.show()

    # plot_corr_scatter(df(X_reduced))
