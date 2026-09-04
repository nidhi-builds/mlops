"""
model_training.py

Stage 4 of the pipeline (dvc stage: model_building).

Trains a RandomForestRegressor to predict manufactured-home PRICE
from the engineered features and pickles the fitted model.
Random forest was chosen over plain linear regression because most
of the features here are one-hot encoded categorical codes, and a
tree-based model handles that kind of feature space (and the
non-linear price effects of SQFT/BEDROOMS) better without needing
extra scaling/interaction terms.
"""

import os
import sys
import pickle

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

from logger import get_logger
from exception import PipelineException

logger = get_logger("model_training")

PARAMS_PATH = "params.yaml"


def load_params(path: str = PARAMS_PATH) -> dict:
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return params["model_building"]
    except Exception as e:
        raise PipelineException(e, sys)


def train_model(train_path: str, target_column: str, model_params: dict) -> RandomForestRegressor:
    try:
        logger.info("Reading training data from %s", train_path)
        train_df = pd.read_csv(train_path)

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        logger.info("Training RandomForestRegressor with params: %s", model_params)
        model = RandomForestRegressor(**model_params)
        model.fit(X_train, y_train)

        logger.info("Model training complete.")
        return model
    except Exception as e:
        raise PipelineException(e, sys)


def save_model(model: RandomForestRegressor, model_path: str):
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved trained model to %s", model_path)
    except Exception as e:
        raise PipelineException(e, sys)


def main():
    try:
        params = load_params()

        model_params = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_split": params["min_samples_split"],
            "min_samples_leaf": params["min_samples_leaf"],
            "random_state": params["random_state"],
        }

        model = train_model(
            train_path=params["train_path"],
            target_column="PRICE",
            model_params=model_params,
        )
        save_model(model, params["model_path"])
        logger.info("Model training stage completed successfully.")
    except Exception as e:
        logger.error(str(e))
        raise


if __name__ == "__main__":
    main()
