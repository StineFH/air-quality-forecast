from src.ingestion.base_ingestion import BaseIngestion
import os
import time
import pandas as pd
from openaq import OpenAQ

class OpenAQIngestion(BaseIngestion):
    def __init__(
        self,
        location_id: int,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> None:
        self.location_id = location_id
        self.date_from   = date_from
        self.date_to     = date_to
        self.client      = None

    def fetch(self) -> pd.DataFrame:
        """Fetch raw data from OpenAQ API and return as DataFrame"""
        self._initialize_client()
        self._verify_location()
        
        # Check existing data to determine start date
        existing_data = self._get_existing_data()
        date_from = None
        if existing_data is not None and not existing_data.empty:
            date_from = existing_data['date'].max().strftime('%Y-%m-%d')
        
        # Fetch measurements
        data = self._fetch_measurements(self.location_id, date_from)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {'date': m.period.datetime_from.utc, 'value': m.value} 
            for m in data
        ])

        df['location_id'] = self.location_id
        return df

    def _verify_location(self) -> None:
        """print the location being used for data collection."""
        location_response = self.client.locations.get(self.location_id)
        location = location_response.results[0]
        
        print(f"Fetching data from location: {location.name} (ID: {self.location_id})")
        if hasattr(location, 'coordinates') and location.coordinates:
            coords = location.coordinates
            print(f"  Coordinates: {coords.latitude}, {coords.longitude}")

    def _initialize_client(self):
        """Initialize the OpenAQ client and find location ID."""
        api_key = os.getenv('OPENAQ_API_KEY')
        if not api_key:
            raise ValueError("OPENAQ_API_KEY environment variable is not set")
        
        self.client = OpenAQ(api_key=api_key)
    
    def _find_pm25_sensor_id(self, location_id: int) -> int:
        """Find a PM2.5 sensor id for the given OpenAQ location_id."""
        sensors_response = self.client.locations.sensors(location_id)
        pm25_sensors = [
            sensor for sensor in sensors_response.results
            if getattr(sensor.parameter, 'name', None) == 'pm25'
        ]
        if not pm25_sensors:
            raise ValueError(
                f"No PM2.5 sensor found for location id={location_id}"
            )
        return pm25_sensors[0].id

    def _fetch_measurements(self, location_id: int, date_from: str | None) -> list:
        """Fetch PM2.5 measurements month by month to avoid API timeouts."""
        sensor_id = self._find_pm25_sensor_id(location_id)

        start = pd.Timestamp(date_from) if date_from else pd.Timestamp(self.date_from or "2023-01-01")
        end   = pd.Timestamp(self.date_to) if self.date_to else pd.Timestamp.now().normalize()

        data = []
        window_start = start
        while window_start < end:
            window_end = min(window_start + pd.DateOffset(months=1), end)
            print(f"  Fetching {window_start.date()} → {window_end.date()} ...", end=" ", flush=True)

            page = 1
            while True:
                try:
                    resp = self.client.measurements.list(
                        sensor_id,
                        datetime_from=window_start.strftime("%Y-%m-%d"),
                        datetime_to=window_end.strftime("%Y-%m-%d"),
                        page=page,
                        limit=1000,
                    )
                except Exception as e:
                    print(f"failed ({e})")
                    break

                measurements = resp.results
                if not measurements:
                    break
                data.extend(measurements)
                print(f"{len(data):,} rows total", end="\r", flush=True)
                if len(measurements) < 1000:
                    break
                page += 1
                time.sleep(3)

            window_start = window_end
            print()
        return data

    def save_raw(self, data: pd.DataFrame) -> None:
        """Save raw data to data/raw/openaq/pm25_<location_id>.csv"""
        path = self._csv_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if os.path.exists(path):
            existing_df = pd.read_csv(path)
            existing_df["date"] = pd.to_datetime(existing_df["date"], utc=True).dt.tz_localize(None)
            data["date"] = pd.to_datetime(data["date"], utc=True).dt.tz_localize(None)
            combined_df = (
                pd.concat([existing_df, data])
                .drop_duplicates(subset=["date"])
                .sort_values("date")
            )
            combined_df.to_csv(path, index=False)
        else:
            data.to_csv(path, index=False)

    def _get_existing_data(self) -> pd.DataFrame | None:
        """Load existing saved data for this location_id if available."""
        path = self._csv_path()
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["date"])
        return None

    def _csv_path(self) -> str:
        return f"data/raw/openaq/pm25_{self.location_id}.csv"
    

if __name__ == "__main__":
    ingestion = OpenAQIngestion(location_id=3050456, date_from="2024-01-01")
    ingestion.fetch_and_save()