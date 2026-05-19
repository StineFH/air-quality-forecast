from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import pandas as pd

from src.ingestion.eea_ingestion import EEAIngestion
from src.ingestion.openaq_ingestion import OpenAQIngestion
from src.cleaning.eea_cleaning import EEACleaner
from src.features.pipeline import FeaturePipeline
# Imported when implemented:
# from src.inference.predict import Predictor

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR  = Path("data/features")

Step = Literal["ingest", "clean", "features", "predict"]

ALL_STEPS: list[Step] = ["clean", "features"]


class Pipeline:
    """
    Orchestrates the full PM2.5 forecasting pipeline.

    Each step reads from and writes to disk so steps can be run independently
    — e.g. the daily GitHub Actions job runs only 'ingest' and 'predict',
    while a full rebuild runs all steps in order.

    Parameters
    ----------
    steps : list of step names to run. Defaults to all steps.
        "ingest"   - fetch latest data from EEA and OpenAQ
        "clean"    - clean raw EEA data and fill gaps
        "features" - build feature matrix from cleaned data
        "predict"  - load latest model and produce tomorrow's forecast

    Usage
    -----
    # Full rebuild
    Pipeline().run()

    # Daily production job (ingest + predict only)
    Pipeline(steps=["ingest", "predict"]).run()

    # Just rebuild features and retrain
    Pipeline(steps=["features"]).run()
    """

    def __init__(self, steps: list[Step] | None = None) -> None:
        self.steps = steps or ALL_STEPS
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)


    def run(self) -> None:
        """Run all configured steps in order."""
        print(f"Pipeline starting — steps: {self.steps}\n{'─' * 50}")
        start = time.time()

        step_map = {
            "ingest": self.run_ingestion,
            "clean": self.run_cleaning,
            "features": self.run_features,
            "predict": self.run_predict,
        }

        for step in self.steps:
            if step not in step_map:
                raise ValueError(f"Unknown step '{step}'. Valid steps: {ALL_STEPS}")
            print(f"\n[{step.upper()}]")
            step_map[step]()

        elapsed = time.time() - start
        print(f"\n{'─' * 50}\nPipeline complete in {elapsed:.1f}s.")


    def run_ingestion(self) -> None:
        """
        Fetch latest data from all sources and save to data/raw/.
        Each ingester appends only new rows (deduplicates on save).
        """
        print("Fetching EEA data...")
        EEAIngestion(
            country_code="DK",
            pollutant="PM2.5",
            aggregation_type="hour",
        ).fetch_and_save()

        print("\nFetching OpenAQ data...")
        OpenAQIngestion(location_id=3050456).fetch_and_save()

    def run_cleaning(self) -> pd.DataFrame:
        """
        Clean raw EEA data, fill gaps, and save to data/processed/pm25_clean.csv.

        Returns the cleaned dataframe (also used internally when running
        cleaning + features together in a single pipeline run).
        """
        print("Cleaning EEA data...")
        df = EEACleaner().clean()

        out_path = PROCESSED_DIR / "pm25_clean.csv"
        df.rename_axis("date").to_csv(out_path)
        print(f"Saved → {out_path}  ({len(df)} rows)")
        return df

    def run_features(self) -> pd.DataFrame:
        """
        Build the feature matrix from cleaned data and save to data/features/.

        Loads from data/processed/pm25_clean.csv so cleaning does not need
        to re-run unless the cleaned data has changed.
        """
        fp = FeaturePipeline()
        return fp.build(self._load_cleaned())

    def run_predict(self) -> None:
        """
        Load the latest trained model and write tomorrow's forecast to
        data/outputs/predictions/.
        """
        # TODO: replace with Predictor once implemented
        raise NotImplementedError(
            "Predictor not yet implemented."
        )

    def _load_cleaned(self) -> pd.DataFrame:
        """Load cleaned data from disk, running cleaning first if not present."""
        path = PROCESSED_DIR / "pm25_clean.csv"
        if not path.exists():
            print("Cleaned data not found — running cleaning step first.")
            return self.run_cleaning()
        return pd.read_csv(path, parse_dates=["date"], index_col="date")