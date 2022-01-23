# Pipeline MLP

## Overview

Node for training a Multilayer Perceptron model given data provided to this function as the time of execution, for making predictions given a pre-trained model and a test data set and for reporting the scores of the predictions performed by previous node.

## Pipeline inputs

| Input | type | Description |
| --- | --- | --- |
`X_train` | `np.array` | Data for training
`y_train` | `np.array` | Predictions for training
`parameters` | `Dict[str, Any]` | Dictionary that contains parameters for MPL model
