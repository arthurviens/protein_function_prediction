import numpy as np
import random


def load_data(path="data/"):
    X = np.loadtxt(path + "protein_train.data")
    y = np.loadtxt(path + "protein_train.solution")

    X_test = np.loadtxt(path + "protein_test.data")
    X_valid = np.loadtxt(path + "protein_valid.data")
    return X, y, X_test, X_valid


def load_shrunk_data(path="data/"):
    X = np.load(path + "shrink_0.npy")
    y = np.load(path + "shrink_1.npy")

    return X, y


def save_model(y_test, y_valid, path=""):
    from zipfile import ZipFile

    np.savetxt(path + "protein_test.predict", y_test, fmt="%d")
    np.savetxt(path + "protein_valid.predict", y_valid, fmt="%d")

    zip_obj = ZipFile("submission.zip", "w")
    zip_obj.write("protein_test.predict")
    zip_obj.write("protein_valid.predict")

    zip_obj.close()


def random_indices(size, percentage):
    list_ = list(range(size))
    random.shuffle(list_)
    return list_[: int(percentage * size)]


def shrink(X, y, percentage=0.1):
    indices = random_indices(X.shape[0], percentage)
    return X[indices, :], y[indices]


def save_data(name, *args, path="data/"):
    for i, arg in enumerate(args):
        np.save(path + name + f"_{i}", arg)
        print(repr(arg), " saved")
