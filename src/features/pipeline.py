from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path

import pandas as pd
import yaml

from src.features.base import BaseFeatureBuilder
from src.features.log_transform import LogTransform


def _discover_builders() -> dict[str, type[BaseFeatureBuilder]]:
    """
    Scan src/features/ and return a registry of {ClassName: class} for every
    concrete subclass of BaseFeatureBuilder. Adding a new feature file requires
    no changes here — drop the file in the directory and it is found automatically.
    """
    registry = {}
    package_path = Path(__file__).parent
    package_name = __name__.rsplit(".", 1)[0]

    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        if module_name in ("pipeline", "base"):
            continue
        module = importlib.import_module(f"{package_name}.{module_name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseFeatureBuilder) and obj is not BaseFeatureBuilder:
                registry[obj.__name__] = obj

    return registry


class FeaturePipeline:
    """
    Builds the full feature matrix by applying an ordered list of
    BaseFeatureBuilder instances.

    Builder classes are auto-discovered from src/features/ — no manual imports
    needed when adding new builders. Configuration is read from config.yaml.

    If LogTransform is configured, it always runs first and target_col is
    updated to log_<target_col> so all subsequent builders derive features
    from the log-transformed series.

    Parameters
    ----------
    config_path : path to config.yaml. Reads the 'features' section.
    """

    def __init__(self, config_path: str | Path = "config/config.yaml") -> None:
        with open(config_path) as f:
            config = yaml.safe_load(f)["features"]

        self.target_col = config["target_col"]
        self._registry = _discover_builders()
        self._builders = self._build_from_config(config)

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        target_col = self.target_col
        feature_cols = []

        for builder in self._builders:
            cols_before = set(df.columns)
            df = builder.build(df, target_col)
            if isinstance(builder, LogTransform):
                target_col = f"log_{target_col}"
            feature_cols.extend(
                c for c in df.columns if c not in cols_before and c != target_col
            )

        n_before = len(df)
        df = df[[target_col, *feature_cols]].dropna()
        print(f"[FeaturePipeline] Dropped {n_before - len(df)} rows with NaN (lag/rolling warmup).")
        print(f"[FeaturePipeline] Feature matrix: {df.shape[0]} rows, {df.shape[1]} columns.")
        return df

    def _build_from_config(self, config: dict) -> list[BaseFeatureBuilder]:
        builders = []
        for entry in config.get("builders", []):
            name = entry["name"]
            if name not in self._registry:
                raise ValueError(
                    f"Builder '{name}' not found. Available: {list(self._registry)}"
                )
            builders.append(
                self._registry[name](**{k: v for k, v in entry.items() if k != "name"})
            )

        builders.sort(key=lambda b: not isinstance(b, LogTransform))
        return builders
