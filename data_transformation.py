import numpy as np
import pandas as pd


def load_data(path="data/"):
    X = np.load(path + "shrink_0.npy")
    y = np.load(path + "shrink_1.npy")

    return X, y


if __name__ == "__main__":
    X, y = load_data()
