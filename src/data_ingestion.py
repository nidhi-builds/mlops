"""
data_ingestion.py

Stage 1 of the pipeline (dvc stage: collection).

Reads the raw Census Manufactured Housing Survey file (puf2022.xls)
and dumps it as a plain csv (dataset/housing.csv) so the rest of the
pipeline doesn't have to keep dealing with Excel.
"""

import os
import sys

import pandas as pd
import yaml

from logger import get_logger
from exception import PipelineException

logger = get_logger("data_ingestion")

PARAMS_PATH = "params.yaml"


def load_params(path: str = PARAMS_PATH) -> dict:
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        logger.info("Loaded params from %s", path)
        return params["data_ingestion"]
    except Exception as e:
        raise PipelineException(e, sys)


def ingest_data(raw_data_path: str, sheet_name: str, ingested_data_path: str) -> pd.DataFrame:
    """Read the raw .xls survey file and save it as a csv."""
    try:
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")

        logger.info("Reading raw data from %s (sheet=%s)", raw_data_path, sheet_name)
        df = pd.read_excel(raw_data_path, sheet_name=sheet_name)
        logger.info("Raw data shape: %s", df.shape)

        os.makedirs(os.path.dirname(ingested_data_path), exist_ok=True)
        df.to_csv(ingested_data_path, index=False)
        logger.info("Saved ingested data to %s", ingested_data_path)

        return df
    except Exception as e:
        raise PipelineException(e, sys)


def main():
    try:
        params = load_params()
        ingest_data(
            raw_data_path=params["raw_data_path"],
            sheet_name=params["sheet_name"],
            ingested_data_path=params["ingested_data_path"],
        )
        logger.info("Data ingestion stage completed successfully.")
    except Exception as e:
        logger.error(str(e))
        raise


if __name__ == "__main__":
    main()
