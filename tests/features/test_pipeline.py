from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.log_transform import LogTransform
from src.features.pipeline import FeaturePipeline


def _daily_df(n: int = 20, start: str = "2020-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({"pm25": rng.uniform(5, 50, n)}, index=idx)


def _write_config(tmp_path: Path, builders: list[dict]) -> Path:
    config = {"features": {"target_col": "pm25", "builders": builders}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return path


class TestFeaturePipeline:
    def test_log_transform_always_runs_first(self, tmp_path):
        # LogTransform is listed last but must sort to position 0.
        config_path = _write_config(tmp_path, [
            {"name": "LagFeatures", "lags": [1]},
            {"name": "LogTransform"},
        ])
        pipeline = FeaturePipeline(config_path)
        assert isinstance(pipeline._builders[0], LogTransform)

    def test_lag_features_use_log_scale_after_log_transform(self, tmp_path):
        config_path = _write_config(tmp_path, [
            {"name": "LogTransform"},
            {"name": "LagFeatures", "lags": [1]},
        ])
        pipeline = FeaturePipeline(config_path)
        result = pipeline.build(_daily_df())
        assert "log_pm25_lag_1" in result.columns
        assert "pm25_lag_1" not in result.columns

    def test_nan_rows_dropped_after_lag_warmup(self, tmp_path):
        config_path = _write_config(tmp_path, [
            {"name": "LagFeatures", "lags": [3]},
        ])
        pipeline = FeaturePipeline(config_path)
        result = pipeline.build(_daily_df())
        assert not result.isnull().any().any()

    def test_output_contains_target_col(self, tmp_path):
        config_path = _write_config(tmp_path, [
            {"name": "LagFeatures", "lags": [1]},
        ])
        pipeline = FeaturePipeline(config_path)
        result = pipeline.build(_daily_df())
        assert "pm25" in result.columns

    def test_output_target_is_log_col_when_log_transform_used(self, tmp_path):
        config_path = _write_config(tmp_path, [
            {"name": "LogTransform"},
            {"name": "LagFeatures", "lags": [1]},
        ])
        pipeline = FeaturePipeline(config_path)
        result = pipeline.build(_daily_df())
        assert "log_pm25" in result.columns
        assert "pm25" not in result.columns

    def test_unknown_builder_raises(self, tmp_path):
        config_path = _write_config(tmp_path, [{"name": "NonExistentBuilder"}])
        with pytest.raises(ValueError, match="not found"):
            FeaturePipeline(config_path)
