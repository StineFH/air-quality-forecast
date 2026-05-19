from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaning.base_cleaning import BaseCleaner
from src.cleaning.missing_data import (
    ConditionalMeanStrategy,
    ExternalSourceStrategy,
    LinearInterpolationStrategy,
    MissingDataHandler,
)
from src.cleaning.transforms import (
    combine_to_hourly_series,
    filter_date_range,
    remove_sensor_errors,
    resample_to_daily,
)
from src.schemas.cleaning import CleanedEEADataset


RAW_PATH               = Path("data/raw/eea/DK_PM25.csv")
OPENAQ_PATH            = Path("data/raw/openaq/pm25_3050456.csv")
START_YEAR             = 2015
SENSOR_ERROR_THRESHOLD = 300.0
MAX_INTERP_DAYS        = 7
OPENAQ_BIAS            = -3.35   # mean bias from overlap analysis (source − primary)
# bias only calculated over 184 so should be recalculated if we get more overlap in the future


class EEACleaner(BaseCleaner):
    """
    Cleans raw EEA PM2.5 hourly data for Aarhus and returns a validated
    daily-mean dataframe conforming to CleanedEEADataset.

    Cleaning steps:
        1. _load_raw()         - parse CSV, set datetime index
        2. _prepare_series()   - filter date range, combine stations, remove sensor errors
        3. _resample()         - hourly → daily mean + observation count
        4. _fill_gaps()        - apply prioritised gap filling:
                                   1. OpenAQ (bias-corrected)
                                   2. Linear interpolation (≤ 7 days)
                                   3. Conditional mean (calendar-day climatology)
    """

    def __init__(
        self,
        raw_path: Path = RAW_PATH,
        openaq_path: Path = OPENAQ_PATH,
    ) -> None:
        super().__init__(raw_path)
        self.openaq_path = openaq_path

    def clean(self) -> pd.DataFrame:
        """
        Run the full cleaning pipeline.

        Returns
        -------
        pd.DataFrame
            Daily index with columns: pm25_mean, n_hourly_obs,
            is_interpolated, is_conditional_mean.
            Conforms to CleanedEEADataset schema.
        """
        data = self._load_raw()
        series = self._prepare_series(data)
        daily = resample_to_daily(series)
        daily = self._fill_gaps(daily)

        self.validate(daily)
        print(
            f"EEA cleaning complete. {len(daily)} daily records "
            f"({daily['is_interpolated'].sum()} linearly interpolated, "
            f"{daily['is_conditional_mean'].sum()} conditional mean)."
        )
        return daily

    def validate(self, df: pd.DataFrame) -> None:
        """Validate output against CleanedEEADataset schema (raises on failure)."""
        CleanedEEADataset.from_dataframe(
            df.reset_index().rename(columns={"index": "date"})
        )

    def _load_raw(self) -> pd.DataFrame:
        """Load raw EEA CSV and return a dataframe with a parsed datetime index."""
        data = pd.read_csv(self.raw_path, parse_dates=["start", "end"])
        data = data.set_index("start").sort_index()
        return data

    def _prepare_series(self, data: pd.DataFrame) -> pd.Series:
        """Apply EEA-specific transforms: date filter, station merge, error removal."""
        data = filter_date_range(data, START_YEAR)
        hourly = combine_to_hourly_series(data)
        return remove_sensor_errors(hourly, threshold=SENSOR_ERROR_THRESHOLD)

    def _load_openaq(self) -> pd.Series | None:
        """Load and daily-resample the OpenAQ backup series if the file exists."""
        if not self.openaq_path.exists():
            return None
        data = pd.read_csv(self.openaq_path, parse_dates=["date"])
        data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_localize(None)
        daily = (
            data.set_index("date")["value"]
            .pipe(remove_sensor_errors, threshold=SENSOR_ERROR_THRESHOLD)
            .resample("D").mean()
        )
        return daily

    def _fill_gaps(self, daily: pd.DataFrame) -> pd.DataFrame:
        strategies = []

        strategies.append(LinearInterpolationStrategy(max_days=MAX_INTERP_DAYS))

        openaq = self._load_openaq()
        if openaq is not None:
            strategies.append(
                ExternalSourceStrategy(source=openaq, bias=OPENAQ_BIAS, label="OpenAQ")
            )

        strategies.append(ConditionalMeanStrategy())

        return MissingDataHandler(strategies).fill(daily)

if __name__ == "__main__":
    cleaner = EEACleaner()
    df = cleaner.clean()
    print(df.head())
