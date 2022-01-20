"""
This is a boilerplate pipeline 'SVC'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node

from .nodes import train_model, predict, report_scores


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                train_model,
                ["X_train_scaled", "y_train", "parameters"],
                "SVC",
                name="train_SVC_model",
            ),
            node(
                predict,
                ["SVC", "X_test_scaled"],
                "SVC_predictions",
                name="predictions_SVC_model",
            ),
            node(
                report_scores,
                ["y_test", "SVC_predictions"],
                None,
                name="report_SVC_model",
            ),
        ]
    )
