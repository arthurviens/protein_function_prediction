# proteins

## Overview

This is a Kedro project, which was generated using `Kedro 0.17.6`.

You can ake a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.


## How to install dependencies

To install them, run:

```shell
pip install -r src/requirements.txt
```

## How to run Kedro project

You can run the Kedro project with:

```shell
kedro run
```

## How to run a pipeline

You can run a Kedro pipeline with:

```shell
kedro run --pipeline=mypipeline
```

Pipelines:
- MLP
- XGBoost

But default, SVC pipeline is chosen.

## How to visualize Kedro pipelines

You need to install `kedro-viz`. It works with `pip` but we encountered some troubles with `conda`.

You can visualize Kedro project with:
```shell
kedro viz
```


## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, `catalog`, and `startup_error`.


### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```shell
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```shell
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```shell
pip install jupyterlab
```

You can also start JupyterLab:

```shell
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```shell
kedro ipython
```

## Notes

### Dependencies

If you encounter some troubles, know that you need at least :
- `kedro` in general
- `numpy` to manipulate data
- `sklearn` for the pipeline _SVC_ and for preprocessing
- `pytorch` for the pipeline _MLP_
- `xgboost` for the pipeline _XGBoost_
- `jupyter` if you want to run the notebook

### Data location

In order to run `kedro`, you must generate a folder `data/` and organize it as the following structure :
```
.
├── 01_raw
│   ├── proteins_X_test.pkl
│   ├── proteins_X_train.pkl
│   ├── proteins_X_valid.pkl
│   ├── proteins_y_train.pkl
├── 02_intermediate
├── 03_primary
├── 04_feature
├── 05_model_input
├── 06_models
├── 07_model_output
└── 08_reporting
```

Don't worry, we create a script for you !

In the main folder of `proteins`, you can find `setup_data.py`. You only need to put data files :
```
proteins
├──  ...
├──  setup_data.py
├──  protein_train.data
├──  protein_train.solution
├──  protein_test.data
├──  protein_valid.data
└──  ...
```
And then run the script.

### Notebook jupyter

In case of you are using a virtual environment, `jupyter` might not know it. For instance, if you are using the python version `3.7`, you should run the following command line to tell to `jupyter` to add a new kernel :
```shell
ipython kernel install --user --name py37 --display-name "Python 3.7"
```
Then you should be fine to run `jupyter` with `kedro` :
```shell
kedro jupyter notebook
```
And don't forget to select the right kernel when you are in the jupyter environment.
