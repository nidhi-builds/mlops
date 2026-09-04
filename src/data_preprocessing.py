"""
data_preprocessing.py

Stage 2 of the pipeline (dvc stage: preprocessing).

The raw survey has a few things that need cleaning before any model
can use it:
  - "jxxxx" columns are just imputation flags added by the Census
    Bureau (0/1 markers for whether a value was imputed) - not useful
    as predictors, so they get dropped.
  - WEIGHT / WGTADJ / CONTROL are survey design fields (sampling
    weights and a row id), not house attributes, so they get dropped
    too.
  - A handful of fields use small sentinel codes (e.g. 9) to mean
    "not reported" instead of an actual NaN. Rows where PRICE or SQFT
    carry one of these codes are removed, since the target/most
    important feature would be meaningless otherwise.
"""

import os
import sys

import pandas as pd
import yaml

from logger import get_logger
from exception import PipelineException

logger = get_logger("data_preprocessing")

PARAMS_PATH = "params.yaml"


def load_params(path: str = PARAMS_PATH) -> dict:
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return params["data_preprocessing"]
    except Exception as e:
        raise PipelineException(e, sys)


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop imputation flag columns and survey-design columns."""
    j_flag_cols = [c for c in df.columns if c.lower().startswith("j")]
    design_cols = [c for c in ["WEIGHT", "WGTADJ", "CONTROL"] if c in df.columns]

    cols_to_drop = j_flag_cols + design_cols
    logger.info("Dropping %d non-predictive columns: %s", len(cols_to_drop), cols_to_drop)

    return df.drop(columns=cols_to_drop, errors="ignore")


def remove_invalid_rows(df: pd.DataFrame, price_min_valid: int, sqft_min_valid: int) -> pd.DataFrame:
    """Drop rows where PRICE/SQFT hold Census 'not reported' sentinel codes."""
    before = df.shape[0]

    df = df[df["PRICE"] >= price_min_valid]
    df = df[df["SQFT"] >= sqft_min_valid]

    after = df.shape[0]
    logger.info("Removed %d rows with invalid/sentinel PRICE or SQFT values", before - after)

    return df.reset_index(drop=True)


def preprocess_data(input_path: str, output_path: str, price_min_valid: int, sqft_min_valid: int) -> pd.DataFrame:
    try:
        logger.info("Reading data from %s", input_path)
        df = pd.read_csv(input_path)
        logger.info("Data shape before preprocessing: %s", df.shape)

        df = drop_irrelevant_columns(df)
        df = remove_invalid_rows(df, price_min_valid, sqft_min_valid)

        # duplicates can creep in from the raw survey export
        dup_count = df.duplicated().sum()
        if dup_count:
            logger.info("Dropping %d duplicate rows", dup_count)
            df = df.drop_duplicates().reset_index(drop=True)

        logger.info("Data shape after preprocessing: %s", df.shape)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Saved preprocessed data to %s", output_path)

        return df
    except Exception as e:
        raise PipelineException(e, sys)


def main():
    try:
        params = load_params()
        preprocess_data(
            input_path=params["input_path"],
            output_path=params["output_path"],
            price_min_valid=params["price_min_valid"],
            sqft_min_valid=params["sqft_min_valid"],
        )
        logger.info("Data preprocessing stage completed successfully.")
    except Exception as e:
        logger.error(str(e))
        raise


if __name__ == "__main__":
    main()
