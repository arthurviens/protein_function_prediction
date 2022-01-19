import pandas as pd 
import numpy as np
import scipy.stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_weights(y):
    weights = np.zeros((y.shape))
    for u in np.unique(y):
        w = 1 / (y[y==u].shape[0] / y.shape[0])
        weights[np.where(y==u)[0]] = w
    assert weights.shape == y.shape, "Shapes are not identical"
    return weights


def describe_histograms(desc, rm_outliers=True):
    for line in desc.index()[1:]:
        values = desc.loc[line]
        #maxvalues = remove_outliers(maxvalues)
        values.hist(bins=50)


def cross_validate_clf(design_matrix, labels, classifier, cv_folds, weights=None, scaling=True, use_PCA = True, rm_out=True):
    pred = np.zeros(labels.shape)
    for i, (tr, te) in enumerate(cv_folds):
        #print(f"{i}/4 cross validation")
        if scaling:
            scaler = preprocessing.StandardScaler()
            scaler.fit(design_matrix[tr, :])
            X_train = scaler.transform(design_matrix[tr, :])
            X_test = scaler.transform(design_matrix[te, :])
            if use_PCA:
                pca = PCA(n_components=900)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
            if rm_out:
                X_train, y_train = remove_outliers(X_train, labels[tr], zscore_th=5, outlier_dim_number=40)
            else:
                y_train = labels[tr]
        else:
            X_train = design_matrix[tr, :]
            X_test = design_matrix[te, :]
            y_train = labels[tr]
        if weights is not None:
            w = weights[tr]
            classifier.fit(X_train, y_train, sample_weight=w)
        else:
            classifier.fit(X_train, y_train)
        pos_idx = list(classifier.classes_).index(1)
        pred[te] = (classifier.predict_proba(X_test))[:, pos_idx]
    return pred


def print_ROC_AUC(y_true, y_pred):
    fpr, tpr, thresholds =  metrics.roc_curve(y_true, y_pred)
    auc_dt = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, '-', 
         label='DT (AUC = %0.2f)' % (np.mean(auc_dt)))

    # Plot the ROC curve
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curves', fontsize=16)
    plt.legend(loc="lower right")
    plt.show()


def remove_outliers(X, y, weights=None, zscore_th=3, outlier_dim_number=5):
    if outlier_dim_number > X.shape[1]:
        outlier_dim_number = 1

    start_shape = X.shape
    z_scores = scipy.stats.zscore(X)

    abs_z_scores = np.abs(z_scores)
    filtered_entries = (1 - (abs_z_scores < zscore_th)).sum(axis=1)
    filtered_entries = 1 - (filtered_entries >= outlier_dim_number)
    X = X[np.where(filtered_entries == 1)]
    y = y[np.where(filtered_entries == 1)]
    print(f"{start_shape[0] - X.shape[0]} outliers deleted out of {start_shape[0]}")
    if weights is not None:
        weights = weights[np.where(filtered_entries == 1)]
        return X, y, weights 
    else:
        return X, y


def visualize_2d3d(data, y, dim, pcaObj):
    explained_var = pcaObj.explained_variance_ratio_
    positive = np.where(y == 0)
    negative = np.where(y != 0)
    if dim == 2:
        fig = plt.figure()
        ax = fig.add_subplot()
        data = data[:, :2].copy()

        ax.scatter(data[positive, 0], data[positive, 1], marker=".", label="Pos class", \
            color = "green")
        ax.scatter(data[negative, 0], data[negative, 1], marker=".", label="Neg class", \
            color = "red")
        plt.legend(loc="upper right")
        plt.xlabel(f"PCA Component 1 ({(explained_var[0] * 100).round(1)}% var)")
        plt.ylabel(f"PCA Component 2 ({(explained_var[1] * 100).round(1)}% var)")
        plt.grid()
        plt.title("Visualization of data on first 2 PCA axis")
        plt.savefig("2d_plot.png")
        #plt.show()
    if dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        data = data[:, :3].copy()

        ax.scatter(data[positive, 0], data[positive, 1], data[positive, 2], marker=".", \
            label="Pos class", color = "green")
        ax.scatter(data[negative, 0], data[negative, 1], data[negative, 2], marker=".", \
            label="Neg class", color = "red")
        ax.set_xlabel(f"PCA Component 1 ({(explained_var[0] * 100).round(1)}% var)")
        ax.set_ylabel(f"PCA Component 2 ({(explained_var[1] * 100).round(1)}% var)")
        ax.set_zlabel(f"PCA Component 3 ({(explained_var[3] * 100).round(1)}% var)")
        plt.grid()
        plt.legend(loc="upper right")
        plt.title("Visualization of data on first 3 PCA axis")
        plt.show()
        #plt.savefig("3d_plot.png")


def visualize_tsne(X, y):
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X)
    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    X_embedded = tsne.fit_transform(X_pca)

    positive = np.where(y == 0)
    negative = np.where(y != 0)
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(X_embedded[positive, 0], X_embedded[positive, 1], marker="o", s=5, \
        label="Pos class", color = "green", alpha=0.5)
    ax.scatter(X_embedded[negative, 0], X_embedded[negative, 1], marker="o", s=5, \
        label="Neg class", color = "red", alpha=0.5)
    plt.legend()
    plt.show()


def remove_hard_corrs(X, X_test=None, X_valid=None, verbose=1):
    shape = X.shape
    corr = pd.DataFrame(X, copy=False).corr()
    upper_corr = pd.DataFrame(np.triu(corr, 1))
    hard_corrs = np.sum(np.abs(upper_corr) > 0.99, axis=1)
    X = np.delete(X, hard_corrs[hard_corrs != 0].index.values, axis=1)
    if verbose > 0:
        print(f"Removed {shape[1] - X.shape[1]} useless columns")
    if (X_test is not None) and (X_valid is not None):
        X_test = np.delete(X_test, hard_corrs[hard_corrs != 0].index.values, axis=1)
        X_valid = np.delete(X_valid, hard_corrs[hard_corrs != 0].index.values, axis=1)
        return X, X_test, X_valid
    else:
        return X


def reduce_var(X):
    pass