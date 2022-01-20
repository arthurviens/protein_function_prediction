"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from proteins.pipelines import split, preprocessing, SVC


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    split_pipeline = split.create_pipeline()
    preprocessing_pipeline = preprocessing.create_pipeline()
    SVC_pipeline = SVC.create_pipeline()

    return {"__default__": split_pipeline + preprocessing_pipeline + SVC_pipeline}
