"""
model_eval.py

Stage 5 of the pipeline (dvc stage: model_evaluation).

Loads the trained model, scores it on the held-out test split, and
writes the metrics to a json file so `dvc metrics show` / `dvc metrics
diff` can pick them up across experiments.
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from logger import get_logger
from exception import PipelineException

logger = get_logger("model_eval")

PARAMS_PATH = "params.yaml"


def load_params(path: str = PARAMS_PATH) -> dict:
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return params["model_evaluation"]
    except Exception as e:
        raise PipelineException(e, sys)


def load_model(model_path: str):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("Loaded model from %s", model_path)
        return model
    except Exception as e:
        raise PipelineException(e, sys)


def evaluate_model(model, test_path: str, target_column: str) -> dict:
    try:
        logger.info("Reading test data from %s", test_path)
        test_df = pd.read_csv(test_path)

        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        predictions = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions))

        metrics = {"rmse": rmse, "mae": mae, "r2_score": r2}
        logger.info("Evaluation metrics: %s", metrics)
        return metrics
    except Exception as e:
        raise PipelineException(e, sys)


def save_metrics(metrics: dict, metrics_path: str):
    try:
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info("Saved metrics to %s", metrics_path)
    except Exception as e:
        raise PipelineException(e, sys)


def main():
    try:
        params = load_params()
        model = load_model(params["model_path"])
        metrics = evaluate_model(model, test_path=params["test_path"], target_column="PRICE")
        save_metrics(metrics, params["metrics_path"])
        logger.info("Model evaluation stage completed successfully.")
    except Exception as e:
        logger.error(str(e))
        raise


if __name__ == "__main__":
    main()
