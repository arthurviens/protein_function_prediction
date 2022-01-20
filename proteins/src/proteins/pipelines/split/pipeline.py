from kedro.pipeline import Pipeline, node

from .nodes import split_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                split_data,
                ["raw_X_train", "raw_y_train", "params:test_data_ratio"],
                list(("X_train", "y_train", "X_test", "y_test")),
                name="split",
            )
        ]
    )
