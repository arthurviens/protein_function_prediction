"""
This is a boilerplate pipeline 'MLP'
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
                "MLP",
                name="train_SVC_model",
                tags="train_SVC_model",
            ),
            node(
                predict,
                ["MLP", "X_test_scaled", "parameters"],
                "MLP_predictions",
                name="predictions_MLP_model",
                tags="train_MLP_model",
            ),
            node(
                report_scores,
                ["y_test", "MLP_predictions"],
                None,
                name="report_MLP_model",
            ),
        ]
    )
