"""
This is a boilerplate pipeline 'XGBoost'
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
                "XGBoost",
                name="train_SVC_model",
                tags="train_SVC_model",
            ),
            node(
                predict,
                ["XGBoost", "X_test_scaled"],
                "XGBoost_predictions",
                name="predictions_XGBoost_model",
                tags="train_XGBoost_model",
            ),
            node(
                report_scores,
                ["y_test", "XGBoost_predictions"],
                None,
                name="report_XGBoost_model",
            ),
        ]
    )