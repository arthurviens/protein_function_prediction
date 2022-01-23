import os
import pickle
from pathlib import Path
import numpy as np


PATH = Path().absolute()

folders = (
    "01_raw",
    "02_intermediate",
    "03_primary",
    "04_feature",
    "05_model_input",
    "06_models",
    "07_model_output",
    "08_reporting",
)

initial_filenames = (
    "protein_train.data",
    "protein_train.solution",
    "protein_test.data",
    "protein_valid.data",
)
final_filenames = (
    "proteins_X_test.pkl",
    "proteins_X_train.pkl",
    "proteins_X_valid.pkl",
    "proteins_y_train.pkl",
)

os.mkdir(PATH / "data")
for folder in folders:
    os.mkdir(PATH / "data" / folder)

alldata = [np.loadtxt(PATH / filename) for filename in initial_filenames]

for filename, data in zip(final_filenames, alldata):
    with open(PATH / "data" / "01_raw" / filename, "wb") as file:
        pickle.dump(data, file)
