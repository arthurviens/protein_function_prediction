import numpy as np
from sklearn import linear_model


def load_data(path="data/"):
    X = np.loadtxt(path + "protein_train.data")
    y = np.loadtxt(path + "protein_train.solution")

    X_test = np.loadtxt(path + "protein_test.data")
    X_valid = np.loadtxt(path + "protein_valid.data")
    return X, y, X_test, X_valid


def save_model(y_test, y_valid, path=""):
    from zipfile import ZipFile

    np.savetxt(path + "protein_test.predict", y_test, fmt="%d")
    np.savetxt(path + "protein_valid.predict", y_valid, fmt="%d")

    zip_obj = ZipFile("submission.zip", "w")
    zip_obj.write("protein_test.predict")
    zip_obj.write("protein_valid.predict")

    zip_obj.close()


if __name__ == "__main__":
    X, y, X_test, X_valid = load_data()

    # Fit and predict

    log_reg = linear_model.LogisticRegression()
    log_reg.fit(X, y)

    # Predict on the test and validation data.
    y_test = log_reg.predict(X_test)
    y_valid = log_reg.predict(X_valid)

    # Save results

    save_model(y_test, y_valid)
