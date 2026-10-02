from ChestCancer.constants import * # CONFIG_FILE_PATH, PARAM_FILE_PATH
from ChestCancer.utils.common import read_yaml, create_directories
from ChestCancer import logger

from ChestCancer.entity.config_entity import DataIngestionConfig


class ConfigManager():

    """
    Manages project configuration by loading YAML files and providing
    component-specific configuration objects.
    """

    def __init__(self, config_filepath = CONFIG_FILE_PATH, param_filepath = PARAM_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.param = read_yaml(param_filepath)

        logger.info("Creating artifacts_root directory")
        create_directories([self.config.artifacts_root]) 


    def get_data_ingestion(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            source_URL= config.source_URL,
            local_data_file= config.local_data_file,
            unzip_dir= config.unzip_dir
        )
    
        return data_ingestion_config


