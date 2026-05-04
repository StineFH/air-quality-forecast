from abc import ABC, abstractmethod
import pandas as pd

class BaseIngestion(ABC):
    
    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Fetch raw data from the source and return as DataFrame"""
        pass
    
    @abstractmethod
    def save_raw(self, data: pd.DataFrame) -> None:
        """Save raw data to data/raw/<source_name>/"""
        pass
    
    def fetch_and_save(self) -> pd.DataFrame:
        """Template method — calls fetch() then save_raw()"""
        data = self.fetch()
        self.save_raw(data)
        return data