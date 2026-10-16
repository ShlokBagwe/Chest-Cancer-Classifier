from ChestCancer.constants import * # CONFIG_FILE_PATH, PARAM_FILE_PATH
from ChestCancer.utils.common import read_yaml, create_directories
from ChestCancer import logger

from ChestCancer.entity.config_entity import DataIngestionConfig
from ChestCancer.entity.config_entity import DataPreparationConfig
from ChestCancer.entity.config_entity import PrepareBaselineModelConfig
from ChestCancer.entity.config_entity import TrainingConfig

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
            root_dir= Path(config.root_dir),
            source_URL= config.source_URL,
            local_data_file= Path(config.local_data_file),
            unzip_dir= Path(config.unzip_dir)
        )
    
        return data_ingestion_config

    
    def get_data_preparation(self) -> DataPreparationConfig:
        
        config = self.config.data_preparation # defined in config.yaml
        params = self.param.data_preparation

        create_directories([config.root_dir])

        data_prepration_config = DataPreparationConfig(
          root_dir = Path(config.root_dir),
          data_dir = Path(config.data_dir),
          prepared_dir= Path(config.prepared_dir),
          
          splits= params.SPLITS,
          label_prefixes= {
            "adenocarcinoma": params.adenocarcinoma,
                "large.cell": params["large.cell"], #accessing using brackets due to '.' present
                "normal": params.normal,
                "squamous.cell": params["squamous.cell"]
          } 
        )

        return data_prepration_config

    
    def get_baseline_model_config(self) -> PrepareBaselineModelConfig:

        config = self.config.prepare_base_model
        params = self.param.baseline_model

        create_directories([config.root_dir])

        baseline_model_config = PrepareBaselineModelConfig(
            root_dir=Path(config.root_dir),
            baseline_model_path=Path(config.baseline_model_path),
            updated_baseline_model_path=Path(config.updated_baseline_model_path),
            params_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
            params_classes=params.CLASSES,
            params_batch_size=params.BATCH_SIZE,
            params_include_top=params.INCLUDE_TOP,
            params_epochs=params.EPOCHS,
            params_weights=params.WEIGHTS,
            params_learning_rate=params.LEARNING_RATE
        )

        return baseline_model_config


    def get_training_config(self) -> TrainingConfig:

        training_config = self.config.training
        prepare_base_model_config = self.config.prepare_base_model
        params = self.param.baseline_model

        create_directories([training_config.root_dir])

        training_config = TrainingConfig(
            root_dir = Path(training_config.root_dir),
            training_model_path= Path(training_config.training_model_path),
            updated_base_model_path= Path(prepare_base_model_config.updated_baseline_model_path),
            training_data= Path(self.config.data_preparation.prepared_dir),
            param_epochs= params.EPOCHS,
            params_patience= params.PATIENCE,
            params_batch_size= params.BATCH_SIZE,
            params_is_augmentation= params.AUGMENTATION,
            params_image_size= params.IMAGE_SIZE,
            params_learning_rate= params.LEARNING_RATE
        )

        return training_config

