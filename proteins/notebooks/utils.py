import pandas as pd 
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os

from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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
    plt.title("TSNE 2D visualization")
    plt.xlabel("TSNE Axis 1")
    plt.ylabel("TSNE Axis 2")
    plt.legend()
    plt.show()
