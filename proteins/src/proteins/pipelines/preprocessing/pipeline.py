from kedro.pipeline import Pipeline, node

from .nodes import scale_data


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                scale_data,
                [
                    "X_train",
                    "X_test",
                    "raw_X_test",
                    "raw_X_valid",
                    "params:scaler_type",
                    "params:n_components_pca",
                ],
                [
                    "X_train_scaled",
                    "X_test_scaled",
                    "X_test_competition",
                    "X_valid_competition",
                ],
                name="preprocessing",
            )
        ]
    )
