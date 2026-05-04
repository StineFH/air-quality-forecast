from src.ingestion.base_ingestion import BaseIngestion
import os
import time
import pandas as pd
from openaq import OpenAQ


class OpenAQIngestion(BaseIngestion):
    def __init__(self, location_id: int):
        """Location IDs can be found on OpenAQ in the end of the url for a location, 
        e.g. https://explore.openaq.org/locations/6236047"""
        self.location_id = location_id
        self.client = None

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
        """Fetch PM2.5 measurements from the API."""
        sensor_id = self._find_pm25_sensor_id(location_id)
        data = []
        limit = 1000
        page = 1
        while True:
            request_kwargs = {
                'page': page,
                'limit': limit,
            }
            if date_from is not None:
                request_kwargs['datetime_from'] = date_from

            measurements_response = self.client.measurements.list(
                sensor_id,
                **request_kwargs,
            )
            measurements = measurements_response.results
            if not measurements:
                break
            data.extend(measurements)
            if len(measurements) < limit:
                break
            page += 1
            time.sleep(2)  # To respect rate limits (60/min, 2000/hour)
        return data

    def save_raw(self, data: pd.DataFrame) -> None:
        """Save raw data to data/raw/openaq/"""
        path = 'data/raw/openaq/pm25_data.csv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if os.path.exists(path):
            existing_df = pd.read_csv(path, parse_dates=['date'])
            combined_df = pd.concat([existing_df, data]).drop_duplicates(subset=['date']).sort_values('date')
            combined_df.to_csv(path, index=False)
        else:
            data.to_csv(path, index=False)

    def _get_existing_data(self) -> pd.DataFrame | None:
        """Helper method to load existing saved data if available."""
        path = 'data/raw/openaq/pm25_data.csv'
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=['date'])
        return None
    

if __name__ == "__main__":
    ingestion = OpenAQIngestion(location_id=6236047)
    ingestion.fetch_and_save()