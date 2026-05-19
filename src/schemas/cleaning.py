from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field, model_validator
from typing import Annotated


class CleanedEEARecord(BaseModel):
    date: pd.Timestamp
    pm25_mean: Annotated[float, Field(ge=0.0, lt=300.0)]  # no None — cleaner guarantees no NaN
    n_hourly_obs: Annotated[int, Field(ge=0, le=24)]
    is_interpolated: bool
    is_conditional_mean: bool

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def interpolated_requires_no_obs(self) -> CleanedEEARecord:
        if self.is_interpolated and self.n_hourly_obs != 0:
            raise ValueError("An interpolated day must have n_hourly_obs = 0")
        return self


class CleanedEEADataset(BaseModel):
    records: list[CleanedEEARecord]

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> CleanedEEADataset:
        required = {"date", "pm25_mean", "n_hourly_obs", "is_interpolated", "is_conditional_mean"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Cleaned dataframe is missing columns: {missing}")
        if df["pm25_mean"].isna().any():
            raise ValueError("pm25_mean contains NaN — cleaner must fill all gaps before validation.")

        records = [
            CleanedEEARecord(**row)
            for row in df[list(required)].to_dict(orient="records")
        ]
        return cls(records=records)