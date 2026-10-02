from ChestCancer.config.configuration import ConfigManager
from ChestCancer.components.data_ingestion import DataIngestion

from ChestCancer import logger

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionPipeline:

    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigManager()
            config_data_ingestion = config.get_data_ingestion()
            
            data_ingestion = DataIngestion(config_data_ingestion)
            data_ingestion.download_file()
            data_ingestion.extract_file()
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f"Stage: {STAGE_NAME} started")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"Stage: {STAGE_NAME} finished")
    except Exception as e:
        logger.exception(e)
        raise e