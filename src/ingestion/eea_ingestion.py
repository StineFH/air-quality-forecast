"""
EEA Air Quality Download Service — PM2.5 ingestion (2013-present).

Covers:
  - E1a verified (dataset=2):  2013-2022  (annually submitted, quality-assured)
  - UTD / E2a  (dataset=1):    2023-now   (near-real-time, unverified)

API docs: https://eeadmz1-downloads-api-appservice.azurewebsites.net/swagger/index.html
"""

from __future__ import annotations

import io
import os
import time

import pandas as pd
import requests

from src.ingestion.base_ingestion import BaseIngestion

_BASE_URL = "https://eeadmz1-downloads-api-appservice.azurewebsites.net"

datasets = {1: 'DATASET_UTD',      # 2023–now   (unverified)
            2: 'DATASET_VERIFIED'} # 2013–2022  (quality-assured)

_DOWNLOAD_DELAY = 0.3


class EEAIngestion(BaseIngestion):
    """
    Fetches PM2.5 (or any pollutant) from the EEA Download Service for 2013-present.

    Parameters
    ----------
    country_code     : ISO 3166-1 alpha-2, e.g. "DK".
    pollutant        : EEA notation, e.g. "PM2.5", "PM10", "NO2".
    city             : Optional EEA city name filter.
    station_filter   : Optional substring matched against sampling-point ID.
    aggregation_type : "hour" (default), "day", or "var".
    email            : Recommended by EEA for anomaly contact.
    """

    def __init__(
        self,
        country_code: str = "DK",
        pollutant: str = "PM2.5",
        city: str | None = None,
        station_filter: str | None = None,
        aggregation_type: str = "hour",
        email: str | None = None,
    ) -> None:
        self.country_code     = country_code.upper()
        self.pollutant        = pollutant
        self.city             = city
        self.station_filter   = station_filter
        self.aggregation_type = aggregation_type
        self.email            = email

    def fetch(self) -> pd.DataFrame:
        """Download both dataset epochs and return a single cleaned DataFrame."""
        frames = []

        for dataset in datasets:
            print(f"\n[EEA] Fetching {datasets[dataset]} (dataset={dataset})")
            df = self._fetch_data(dataset)
            if df is not None and not df.empty:
                frames.append(df)
                print(f"{len(df):,} rows")
            else:
                print(f"  — no data returned")

        if not frames:
            print("[EEA] No data found.")
            return pd.DataFrame()

        combined = (
            pd.concat(frames, ignore_index=True)
            .sort_values("start")
            .reset_index(drop=True)
        )
        print(f"\n[EEA] Total: {len(combined):,} rows across {combined['station_id'].nunique()} station(s).")
        return combined

    def save_raw(self, data: pd.DataFrame) -> None:
        """
        Append-save to data/raw/eea/<COUNTRY>_<pollutant>.csv.
        Deduplicates on station_id + start. Station ID is preserved as a column.
        """
        if data.empty:
            print("[EEA] Nothing to save — DataFrame is empty.")
            return

        path = self._csv_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data["start"] = pd.to_datetime(data["start"], utc=True)

        if os.path.exists(path):
            existing = pd.read_csv(path, parse_dates=["start"])
            combined = (
                pd.concat([existing, data])
                .drop_duplicates(subset=["station_id", "start"])
                .sort_values(["station_id", "start"])
            )
            combined.to_csv(path, index=False)
            print(f"[EEA] Updated {path}  ({len(combined):,} rows total)")
        else:
            data.to_csv(path, index=False)
            print(f"[EEA] Created {path}  ({len(data):,} rows)")

    def _request(self, method: str, url: str, **kwargs) -> requests.Response | None:
        """Shared HTTP request with unified error handling."""
        try:
            resp = requests.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as e:
            content_type = e.response.headers.get("Content-Type", "")
            detail = e.response.json() if "application/json" in content_type else e.response.text.strip()
            print(f"\n  [WARN] HTTP {e.response.status_code} for {url}: {detail}")
        except requests.RequestException as e:
            print(f"\n  [WARN] Request failed for {url}: {e}")
        return None

    def _build_request_body(self, dataset: int) -> dict:
        body: dict = {
            "countries":       [self.country_code],
            "cities":          [self.city] if self.city else [],
            "pollutants":      [self.pollutant],
            "dataset":         dataset,
            "aggregationType": self.aggregation_type,
            "source":          "Custom script",
        }
        if self.email:
            body["email"] = self.email
        return body

    def _get_parquet_urls(self, dataset: int) -> list[str]:
        endpoint = f"{_BASE_URL}/ParquetFile/urls"
        resp = self._request("POST", endpoint, json=self._build_request_body(dataset), timeout=60)
        if resp is None:
            return []
        return [u.strip() for u in resp.text.split("\n")[1:] if u.strip()]

    def _fetch_data(self, dataset: int) -> pd.DataFrame | None:
        urls = self._get_parquet_urls(dataset)
        if not urls:
            return None

        print(f"Found {len(urls)} parquet file(s).")
        dfs = []
        for idx, url in enumerate(urls, start=1):
            print(f"[{idx}/{len(urls)}] {url.split('/')[-1]} ", end="", flush=True)
            raw = self._download_parquet_url(url)
            if raw is not None:
                dfs.append(raw)
                print(f"{len(raw):,} rows")
            else:
                print("skipped")
            if idx < len(urls):
                time.sleep(_DOWNLOAD_DELAY)

        return self._normalize(pd.concat(dfs, ignore_index=True)) if dfs else None

    def _download_parquet_url(self, url: str) -> pd.DataFrame | None:
        resp = self._request("GET", url, timeout=120)
        if resp is None:
            return None
        try:
            return pd.read_parquet(io.BytesIO(resp.content))
        except Exception as e:
            print(f"\n  [WARN] Could not parse parquet from {url}: {e}")
            return None

    def _normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns={
            "Samplingpoint": "station_id",
            "Start":         "start",
            "End":           "end",
            "Value":         "value",
            "Unit":          "unit",
            "AggType":       "agg_type",
        })
        df = df.drop(columns=["FkObservationLog", "ResultTime", "Pollutant",
                              "DataCapture", "Validity", "Verification"], errors="ignore")
        df["start"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
        df["end"]   = pd.to_datetime(df["end"],   utc=True, errors="coerce")
        if self.station_filter:
            df = df[df["station_id"].str.contains(self.station_filter, na=False)]
        return df.reset_index(drop=True)

    def _csv_path(self) -> str:
        slug = f"{self.country_code}_{self.pollutant.replace('.', '').replace(' ', '_')}"
        return f"data/raw/eea/{slug}.csv"

    def _get_existing_data(self) -> pd.DataFrame | None:
        path = self._csv_path()
        return pd.read_csv(path, parse_dates=["start"]) if os.path.exists(path) else None


if __name__ == "__main__":
    # ingestion = EEAIngestion(
    #     country_code="DK",
    #     pollutant="PM2.5",
    #     aggregation_type="hour",
    # )
    # ingestion.fetch_and_save()

    # Fetch from Swedish station in Malmø to fill out missing data in 2023 December
    ingestion = EEAIngestion(
        country_code="SE",
        pollutant="PM2.5",
        aggregation_type="hour",
        station_filter='SPO-SE0001A_06001_100'
    )
    ingestion.fetch_and_save()

