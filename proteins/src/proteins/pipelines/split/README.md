# Pipeline split

## Overview

Node for splitting data set into training and test sets, each split into features and labels

## Pipeline inputs

| Input | type | Description |
| --- | --- | --- |
`X` | `np.array` | samples
`y` | `np.array` | predictions
`test_ratio` | `float` | percentage of split ratio

## Pipeline outputs

| Output | type | Description |
| --- | --- | --- |
`X_train` | `np.array`| Data for training
`y_train` | `np.array`| Predictions for testing
`X_test` | `np.array`| Data for testing
`y_test` | `np.array`| Predictions for testing
