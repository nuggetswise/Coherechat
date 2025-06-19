"""
Flexible data source connectors for compensation planning.
Supports multiple data formats and sources with a unified interface.
"""
import os
import pandas as pd
import json
import requests
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import uuid

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comp_data_sources")

class CompensationDataSource:
    """Base class for all compensation data sources."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.last_refresh = None
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get compensation data as a list of dictionaries."""
        raise NotImplementedError("Subclasses must implement get_data()")
    
    def refresh(self) -> bool:
        """Refresh data from source."""
        try:
            self.last_refresh = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Error refreshing {self.name}: {e}")
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about this data source."""
        return {
            "name": self.name,
            "description": self.description,
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "record_count": len(self.get_data())
        }

class CSVDataSource(CompensationDataSource):
    """Load compensation data from CSV files."""
    
    def __init__(self, file_path: str, name: str = None, description: str = ""):
        self.file_path = file_path
        name = name or os.path.basename(file_path)
        super().__init__(name, description or f"CSV data from {os.path.basename(file_path)}")
        self._data = None
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get compensation data from CSV."""
        if self._data is None:
            self.refresh()
        return self._data or []
    
    def refresh(self) -> bool:
        """Reload data from CSV file."""
        try:
            if not os.path.exists(self.file_path):
                logger.error(f"CSV file not found: {self.file_path}")
                return False
            
            df = pd.read_csv(self.file_path)
            self._data = df.to_dict(orient="records")
            super().refresh()
            logger.info(f"Loaded {len(self._data)} records from {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            self._data = []
            return False

class JSONDataSource(CompensationDataSource):
    """Load compensation data from JSON files."""
    
    def __init__(self, file_path: str, name: str = None, description: str = ""):
        self.file_path = file_path
        name = name or os.path.basename(file_path)
        super().__init__(name, description or f"JSON data from {os.path.basename(file_path)}")
        self._data = None
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get compensation data from JSON."""
        if self._data is None:
            self.refresh()
        return self._data or []
    
    def refresh(self) -> bool:
        """Reload data from JSON file."""
        try:
            if not os.path.exists(self.file_path):
                logger.error(f"JSON file not found: {self.file_path}")
                return False
            
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                
            # Handle both array and object formats
            if isinstance(data, list):
                self._data = data
            elif isinstance(data, dict) and "data" in data:
                self._data = data["data"]
            else:
                self._data = [data]
                
            super().refresh()
            logger.info(f"Loaded {len(self._data)} records from {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading JSON: {e}")
            self._data = []
            return False

class APIDataSource(CompensationDataSource):
    """Load compensation data from a REST API."""
    
    def __init__(self, api_url: str, headers: Dict[str, str] = None, params: Dict[str, str] = None, 
                 name: str = None, description: str = ""):
        self.api_url = api_url
        self.headers = headers or {}
        self.params = params or {}
        name = name or f"API-{api_url.split('/')[-1]}"
        super().__init__(name, description or f"API data from {api_url}")
        self._data = None
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get compensation data from API."""
        if self._data is None:
            self.refresh()
        return self._data or []
    
    def refresh(self) -> bool:
        """Reload data from API."""
        try:
            response = requests.get(self.api_url, headers=self.headers, params=self.params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle both array and object formats
            if isinstance(data, list):
                self._data = data
            elif isinstance(data, dict) and "data" in data:
                self._data = data["data"]
            else:
                self._data = [data]
                
            super().refresh()
            logger.info(f"Loaded {len(self._data)} records from {self.api_url}")
            return True
        except Exception as e:
            logger.error(f"Error loading from API: {e}")
            self._data = []
            return False

class ExcelDataSource(CompensationDataSource):
    """Load compensation data from Excel files."""
    
    def __init__(self, file_path: str, sheet_name: str = 0, name: str = None, description: str = ""):
        self.file_path = file_path
        self.sheet_name = sheet_name
        name = name or os.path.basename(file_path)
        super().__init__(name, description or f"Excel data from {os.path.basename(file_path)}")
        self._data = None
    
    def get_data(self) -> List[Dict[str, Any]]:
        """Get compensation data from Excel."""
        if self._data is None:
            self.refresh()
        return self._data or []
    
    def refresh(self) -> bool:
        """Reload data from Excel file."""
        try:
            if not os.path.exists(self.file_path):
                logger.error(f"Excel file not found: {self.file_path}")
                return False
            
            df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            self._data = df.to_dict(orient="records")
            super().refresh()
            logger.info(f"Loaded {len(self._data)} records from {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading Excel: {e}")
            self._data = []
            return False

class CompensationDataManager:
    """Manager for multiple compensation data sources."""
    
    def __init__(self):
        self.sources = {}
        self.default_source = None
    
    def add_source(self, source: CompensationDataSource, is_default: bool = False) -> bool:
        """Add a data source to the manager."""
        try:
            self.sources[source.name] = source
            if is_default:
                self.default_source = source.name
            return True
        except Exception as e:
            logger.error(f"Error adding source {source.name}: {e}")
            return False
    
    def remove_source(self, source_name: str) -> bool:
        """Remove a data source from the manager."""
        if source_name in self.sources:
            del self.sources[source_name]
            if self.default_source == source_name:
                self.default_source = next(iter(self.sources)) if self.sources else None
            return True
        return False
    
    def get_source(self, source_name: str = None) -> Optional[CompensationDataSource]:
        """Get a specific data source or the default one."""
        if source_name and source_name in self.sources:
            return self.sources[source_name]
        if self.default_source:
            return self.sources[self.default_source]
        return None
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get combined data from all sources."""
        all_data = []
        for source in self.sources.values():
            all_data.extend(source.get_data())
        return all_data
    
    def refresh_all(self) -> Dict[str, bool]:
        """Refresh all data sources."""
        results = {}
        for name, source in self.sources.items():
            results[name] = source.refresh()
        return results
    
    def get_source_names(self) -> List[str]:
        """Get names of all registered data sources."""
        return list(self.sources.keys())
    
    def get_sources_metadata(self) -> List[Dict[str, Any]]:
        """Get metadata about all sources."""
        return [source.get_metadata() for source in self.sources.values()]

# Initialize the global data manager
data_manager = CompensationDataManager()

# Helper function to set up default data sources
def setup_default_data_sources():
    """Set up default data sources from local files and configured APIs."""
    # Use only the data directory as specified
    data_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    if data_dir.exists() and data_dir.is_dir():
        # First look for compensation data file to set as default
        comp_data_path = data_dir / "Compensation Data.csv"
        if comp_data_path.exists():
            csv_source = CSVDataSource(
                file_path=str(comp_data_path),
                name="Default Compensation Data",
                description="Internal compensation benchmark data"
            )
            data_manager.add_source(csv_source, is_default=True)
            logger.info(f"Set default data source: {comp_data_path}")
        
        # Add all CSV files
        for csv_file in data_dir.glob("*.csv"):
            # Skip if we already added the compensation data file
            if csv_file.name == "Compensation Data.csv" and comp_data_path.exists():
                continue
                
            csv_source = CSVDataSource(
                file_path=str(csv_file),
                name=f"CSV: {csv_file.stem}",
                description=f"Compensation data from {csv_file.name}"
            )
            data_manager.add_source(csv_source)
            
            # If we don't have a default source yet, use this as default
            if not data_manager.default_source:
                data_manager.default_source = csv_source.name
                logger.info(f"Set default data source: {csv_file}")
        
        # Add JSON files
        for json_file in data_dir.glob("*.json"):
            json_source = JSONDataSource(
                file_path=str(json_file),
                name=f"JSON: {json_file.stem}",
                description=f"Compensation data from {json_file.name}"
            )
            data_manager.add_source(json_source)
        
        # Add Excel files
        for excel_file in data_dir.glob("*.xlsx"):
            excel_source = ExcelDataSource(
                file_path=str(excel_file),
                name=f"Excel: {excel_file.stem}",
                description=f"Compensation data from {excel_file.name}"
            )
            data_manager.add_source(excel_source)
    
    # If no sources were added, log a warning
    if not data_manager.sources:
        logger.warning(f"No compensation data sources were found in {data_dir} directory.")
        logger.info(f"Please ensure compensation data files are placed in the {data_dir} directory.")
        
    return data_manager

# Initialize default sources when module is imported
setup_default_data_sources()

def get_compensation_data_sources() -> List[Dict[str, Any]]:
    """Get all available compensation data sources"""
    
    # Initialize data sources list
    data_sources = []
    
    # Get the data directory path
    data_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
    
    # Check if directory exists
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return data_sources
    
    # Look for CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        try:
            # Try to read the CSV file
            df = pd.read_csv(csv_file)
            
            # Check if it's a compensation dataset
            if all(col in df.columns for col in ['job_title', 'base_salary_usd']):
                # Create data source metadata
                data_source = {
                    "id": str(uuid.uuid4()),
                    "name": csv_file.stem,
                    "type": "compensation_data",
                    "path": str(csv_file),
                    "format": "csv",
                    "record_count": len(df),
                    "last_updated": datetime.fromtimestamp(csv_file.stat().st_mtime).isoformat(),
                    "columns": df.columns.tolist(),
                    "sample": df.head(3).to_dict(orient='records')
                }
                
                # Add statistics if available
                if 'base_salary_usd' in df.columns:
                    data_source["statistics"] = {
                        "avg_base_salary": int(df['base_salary_usd'].mean()),
                        "min_base_salary": int(df['base_salary_usd'].min()),
                        "max_base_salary": int(df['base_salary_usd'].max()),
                        "unique_roles": len(df['job_title'].unique()) if 'job_title' in df.columns else 0,
                        "unique_locations": len(df['location'].unique()) if 'location' in df.columns else 0
                    }
                
                data_sources.append(data_source)
                logger.info(f"Found compensation data source: {csv_file.name} with {len(df)} records")
        except Exception as e:
            logger.error(f"Error processing CSV file {csv_file}: {str(e)}")
    
    # If no compensation data sources found
    if not data_sources:
        logger.warning(f"No compensation data sources were found in {data_dir} directory.")
        logger.info(f"Please ensure compensation data files are placed in the {data_dir} directory.")
    
    return data_sources