"""
test_pipeline.py

Integration tests for the DVC pipeline defined in dvc.yaml / params.yaml:

    collection -> preprocessing -> features -> model_building -> model_evaluation

These tests assume the pipeline has already been executed at least once
(`dvc repro` or `dvc exp run`) so that the stage outputs exist on disk.
Each stage's tests are skipped (not failed) if that stage's output is
missing, so this file is safe to run at any point in development.

Run with:
    pytest test_pipeline.py -v
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# Shared fixtures / config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def params():
    """Load params.yaml once for the whole test session."""
    path = ROOT / "params.yaml"
    assert path.exists(), "params.yaml not found at repo root"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _skip_if_missing(path: Path):
    if not path.exists():
        pytest.skip(f"Expected pipeline output not found: {path} "
                     f"(run `dvc repro` first)")


# ---------------------------------------------------------------------------
# params.yaml / dvc.yaml sanity checks
# ---------------------------------------------------------------------------

class TestConfig:
    REQUIRED_SECTIONS = [
        "data_ingestion",
        "data_preprocessing",
        "feature_engineering",
        "model_building",
        "model_evaluation",
    ]

    def test_params_has_all_stage_sections(self, params):
        for section in self.REQUIRED_SECTIONS:
            assert section in params, f"Missing '{section}' section in params.yaml"

    def test_data_ingestion_params(self, params):
        di = params["data_ingestion"]
        assert di["raw_data_path"].endswith((".xls", ".xlsx"))
        assert di["sheet_name"]
        assert di["ingested_data_path"].endswith(".csv")

    def test_preprocessing_thresholds_are_sane(self, params):
        dp = params["data_preprocessing"]
        assert dp["price_min_valid"] > 0
        assert dp["sqft_min_valid"] > 0

    def test_feature_engineering_split_config(self, params):
        fe = params["feature_engineering"]
        assert fe["target_column"] == "PRICE"
        assert 0 < fe["test_size"] < 1

    def test_model_building_hyperparams(self, params):
        mb = params["model_building"]
        assert mb["n_estimators"] > 0
        assert mb["max_depth"] > 0
        assert mb["min_samples_split"] >= 2
        assert mb["min_samples_leaf"] >= 1

    def test_dvc_yaml_exists_and_parses(self):
        path = ROOT / "dvc.yaml"
        assert path.exists(), "dvc.yaml not found at repo root"
        with open(path, "r") as f:
            spec = yaml.safe_load(f)
        expected_stages = {
            "collection",
            "preprocessing",
            "features",
            "model_building",
            "model_evaluation",
        }
        assert expected_stages.issubset(spec["stages"].keys())


# ---------------------------------------------------------------------------
# Stage 1: collection (data_ingestion.py)
# ---------------------------------------------------------------------------

class TestCollectionStage:
    def test_ingested_csv_exists_and_is_nonempty(self, params):
        out_path = ROOT / params["data_ingestion"]["ingested_data_path"]
        _skip_if_missing(out_path)
        df = pd.read_csv(out_path)
        assert not df.empty, "Ingested housing.csv is empty"

    def test_ingested_csv_has_price_column(self, params):
        out_path = ROOT / params["data_ingestion"]["ingested_data_path"]
        _skip_if_missing(out_path)
        df = pd.read_csv(out_path)
        assert any(col.upper() == "PRICE" for col in df.columns), (
            "Ingested data is missing a PRICE column"
        )


# ---------------------------------------------------------------------------
# Stage 2: preprocessing (data_preprocessing.py)
# ---------------------------------------------------------------------------

class TestPreprocessingStage:
    @pytest.fixture
    def processed_df(self, params):
        out_path = ROOT / params["data_preprocessing"]["output_path"]
        _skip_if_missing(out_path)
        return pd.read_csv(out_path)

    def test_processed_csv_nonempty(self, processed_df):
        assert not processed_df.empty

    def test_no_nulls_in_target_column(self, processed_df, params):
        target = params["feature_engineering"]["target_column"]
        if target in processed_df.columns:
            assert processed_df[target].isnull().sum() == 0

    def test_price_respects_min_threshold(self, processed_df, params):
        min_price = params["data_preprocessing"]["price_min_valid"]
        price_col = next(
            (c for c in processed_df.columns if c.upper() == "PRICE"), None
        )
        if price_col is None:
            pytest.skip("No PRICE column found in processed data")
        assert (processed_df[price_col] >= min_price).all(), (
            f"Found rows with {price_col} below configured minimum ({min_price})"
        )

    def test_sqft_respects_min_threshold(self, processed_df, params):
        min_sqft = params["data_preprocessing"]["sqft_min_valid"]
        sqft_col = next(
            (c for c in processed_df.columns if "SQFT" in c.upper()), None
        )
        if sqft_col is None:
            pytest.skip("No SQFT-like column found in processed data")
        assert (processed_df[sqft_col] >= min_sqft).all(), (
            f"Found rows with {sqft_col} below configured minimum ({min_sqft})"
        )

    def test_row_count_did_not_increase_vs_raw(self, processed_df, params):
        raw_path = ROOT / params["data_ingestion"]["ingested_data_path"]
        _skip_if_missing(raw_path)
        raw_df = pd.read_csv(raw_path)
        assert len(processed_df) <= len(raw_df), (
            "Preprocessing should only filter rows, not add them"
        )


# ---------------------------------------------------------------------------
# Stage 3: features (feature_engg.py)
# ---------------------------------------------------------------------------

class TestFeatureEngineeringStage:
    @pytest.fixture
    def train_test(self, params):
        fe = params["feature_engineering"]
        train_path = ROOT / fe["train_path"]
        test_path = ROOT / fe["test_path"]
        _skip_if_missing(train_path)
        _skip_if_missing(test_path)
        return pd.read_csv(train_path), pd.read_csv(test_path)

    def test_train_and_test_nonempty(self, train_test):
        train_df, test_df = train_test
        assert not train_df.empty
        assert not test_df.empty

    def test_train_test_split_ratio(self, train_test, params):
        train_df, test_df = train_test
        total = len(train_df) + len(test_df)
        actual_test_ratio = len(test_df) / total
        expected_ratio = params["feature_engineering"]["test_size"]
        assert abs(actual_test_ratio - expected_ratio) < 0.03, (
            f"Test split ratio {actual_test_ratio:.3f} deviates from "
            f"configured {expected_ratio}"
        )

    def test_train_and_test_have_same_columns(self, train_test):
        train_df, test_df = train_test
        assert list(train_df.columns) == list(test_df.columns)

    def test_target_column_present(self, train_test, params):
        train_df, test_df = train_test
        target = params["feature_engineering"]["target_column"]
        assert target in train_df.columns
        assert target in test_df.columns

    def test_no_overlap_between_train_and_test(self, train_test):
        train_df, test_df = train_test
        merged = pd.merge(train_df, test_df, how="inner")
        assert merged.empty, "Train and test sets overlap"


# ---------------------------------------------------------------------------
# Stage 4: model_building (model_training.py)
# ---------------------------------------------------------------------------

class TestModelBuildingStage:
    @pytest.fixture
    def model(self, params):
        model_path = ROOT / params["model_building"]["model_path"]
        _skip_if_missing(model_path)
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def test_model_loads(self, model):
        assert model is not None

    def test_model_has_predict_method(self, model):
        assert hasattr(model, "predict")

    def test_model_hyperparams_match_config(self, model, params):
        mb = params["model_building"]
        if hasattr(model, "n_estimators"):
            assert model.n_estimators == mb["n_estimators"]
        if hasattr(model, "max_depth"):
            assert model.max_depth == mb["max_depth"]
        if hasattr(model, "min_samples_split"):
            assert model.min_samples_split == mb["min_samples_split"]
        if hasattr(model, "min_samples_leaf"):
            assert model.min_samples_leaf == mb["min_samples_leaf"]

    def test_model_predicts_on_train_features(self, model, params):
        train_path = ROOT / params["model_building"]["train_path"]
        _skip_if_missing(train_path)
        train_df = pd.read_csv(train_path)
        target = params["feature_engineering"]["target_column"]
        # Keep all non-target columns (incl. one-hot/bool dummy columns like
        # BEDROOMS_3, FOOTINGS_2) — filtering by dtype drops bool/uint8
        # columns and desyncs from the feature set the model was fit on.
        X = train_df.drop(columns=[target], errors="ignore")
        preds = model.predict(X)
        assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# Stage 5: model_evaluation (model_eval.py)
# ---------------------------------------------------------------------------

class TestModelEvaluationStage:
    @pytest.fixture
    def metrics(self, params):
        metrics_path = ROOT / params["model_evaluation"]["metrics_path"]
        _skip_if_missing(metrics_path)
        with open(metrics_path, "r") as f:
            return json.load(f)

    def test_metrics_file_is_valid_json(self, metrics):
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_metrics_are_numeric(self, metrics):
        for key, value in metrics.items():
            assert isinstance(value, (int, float)), (
                f"Metric '{key}' is not numeric: {value!r}"
            )

    def test_r2_metric_within_valid_range(self, metrics):
        r2_key = next((k for k in metrics if "r2" in k.lower()), None)
        if r2_key is None:
            pytest.skip("No R2-style metric found in metrics.json")
        assert metrics[r2_key] <= 1.0, "R2 score cannot exceed 1.0"

    def test_error_metrics_are_non_negative(self, metrics):
        for key, value in metrics.items():
            if any(tag in key.lower() for tag in ("mae", "mse", "rmse")):
                assert value >= 0, f"Error metric '{key}' should be non-negative"


# ---------------------------------------------------------------------------
# End-to-end: model evaluated on the held-out test set matches reported metrics
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_model_r2_on_test_set_is_reasonable(self, params):
        """
        Sanity check that the trained model beats a naive mean-baseline
        on the held-out test set (i.e. it has actually learned something).
        """
        model_path = ROOT / params["model_building"]["model_path"]
        test_path = ROOT / params["feature_engineering"]["test_path"]
        _skip_if_missing(model_path)
        _skip_if_missing(test_path)

        from sklearn.metrics import r2_score

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        test_df = pd.read_csv(test_path)
        target = params["feature_engineering"]["target_column"]
        # Keep all non-target columns — see note in TestModelBuildingStage.
        X_test = test_df.drop(columns=[target], errors="ignore")
        y_test = test_df[target]

        preds = model.predict(X_test)
        score = r2_score(y_test, preds)

        assert score > 0, (
            f"Model R2 on test set ({score:.3f}) does not beat a "
            f"naive mean-baseline — check training/feature pipeline"
        )