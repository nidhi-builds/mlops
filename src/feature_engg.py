"""
feature_engg.py

Stage 3 of the pipeline (dvc stage: features).

Turns the cleaned survey rows into a model-ready table:
  - SHIPMONTH comes in as YYYYMM (e.g. 202203) - only the month part
    actually varies within the survey year, so it's reduced to a
    1-12 "SHIP_MONTH" number.
  - The remaining coded fields (REGION, LOCATION, FOOTINGS, PIERS,
    SECURED, TITLED, LEASE, FINALDEST, STATUS, SECTIONS, BEDROOMS)
    are categorical codes, not real numbers, so they're one-hot
    encoded rather than fed to the model as-is.
  - SQFT is left as a numeric feature.
  - Finally the data is split into train/test sets for the next
    stages.
"""

import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from logger import get_logger
from exception import PipelineException

logger = get_logger("feature_engg")

PARAMS_PATH = "params.yaml"

CATEGORICAL_COLUMNS = [
    "STATUS", "FINALDEST", "FOOTINGS", "LEASE", "LOCATION",
    "REGION", "PIERS", "SECURED", "TITLED", "SECTIONS", "BEDROOMS",
]
NUMERIC_COLUMNS = ["SQFT", "SHIP_MONTH"]


def load_params(path: str = PARAMS_PATH) -> dict:
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return params["feature_engineering"]
    except Exception as e:
        raise PipelineException(e, sys)


def build_features(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    try:
        df = df.copy()

        # collapse YYYYMM -> month number
        df["SHIP_MONTH"] = (df["SHIPMONTH"] % 100).astype(int)
        df = df.drop(columns=["SHIPMONTH"])

        cat_cols_present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
        logger.info("One-hot encoding categorical columns: %s", cat_cols_present)
        df = pd.get_dummies(df, columns=cat_cols_present, drop_first=True)

        # keep target column at the end, everything else is a feature
        feature_cols = [c for c in df.columns if c != target_column]
        df = df[feature_cols + [target_column]]

        logger.info("Feature matrix shape after encoding: %s", df.shape)
        return df
    except Exception as e:
        raise PipelineException(e, sys)


def split_and_save(df: pd.DataFrame, train_path: str, test_path: str, test_size: float, random_state: int):
    try:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        logger.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)

        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_path), exist_ok=True)

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info("Saved train data to %s and test data to %s", train_path, test_path)
    except Exception as e:
        raise PipelineException(e, sys)


def main():
    try:
        params = load_params()

        logger.info("Reading preprocessed data from %s", params["input_path"])
        df = pd.read_csv(params["input_path"])

        features_df = build_features(df, target_column=params["target_column"])

        split_and_save(
            features_df,
            train_path=params["train_path"],
            test_path=params["test_path"],
            test_size=params["test_size"],
            random_state=params["random_state"],
        )
        logger.info("Feature engineering stage completed successfully.")
    except Exception as e:
        logger.error(str(e))
        raise


if __name__ == "__main__":
    main()
