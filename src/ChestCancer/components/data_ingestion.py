from ChestCancer.utils.common import get_size
import zipfile
import gdown
import os
from ChestCancer import logger
from ChestCancer.entity.config_entity import DataIngestionConfig


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):

        try:
            data_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)
            logger.info(f"Downloading data from {data_url} to {zip_download_dir}")

            file_id = self.config.source_URL.split("/")[-2]
            url = f"https://drive.google.com/uc?id={file_id}"

            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            output_zip = "artifacts/data_ingestion/data.zip"

            gdown.download(url, output_zip, quiet=False)

        except Exception as e:
            raise e

    
    def extract_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip:
            zip.extractall(unzip_path)



