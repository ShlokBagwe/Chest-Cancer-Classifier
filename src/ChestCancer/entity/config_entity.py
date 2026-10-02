from dataclasses import dataclass # No need of constructor for a class
from pathlib import Path

@dataclass( frozen=True) # No changes can be made externally
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path