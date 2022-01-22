# Pipeline preprocessing

## Overview

Scale data and apply a PCA on it

## Pipeline inputs

| Input | type | Description |
| --- | --- | --- |
`X_train`| `np.array`| train data set
`X_test`| `np.array`| test data set
`X_test_competition` | `np.array`| data for competition
`X_valid`| `np.array` | data for competition
`scaler`| `str`  | only `"minmax"` or `"standard"` are implemented
`n_components`| `float`  | between 0 and 1

## Pipeline outputs

| Output | type | Description |
| --- | --- | --- |
`X_train` | `np.array`| Data for training
`X_test` | `np.array`|  Data for testing
`X_test_competition` | `np.array`| Data for competition purpose
`X_valid` | `np.array`| Data for competition purpose
