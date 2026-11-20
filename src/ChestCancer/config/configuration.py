from ChestCancer.constants import * # CONFIG_FILE_PATH, PARAM_FILE_PATH
from ChestCancer.utils.common import read_yaml, create_directories
from ChestCancer import logger

from ChestCancer.entity.config_entity import DataIngestionConfig
from ChestCancer.entity.config_entity import DataPreparationConfig
from ChestCancer.entity.config_entity import PrepareBaselineModelConfig
from ChestCancer.entity.config_entity import TrainingConfig
from ChestCancer.entity.config_entity import EvaluationConfig


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
            params_augmentation=bool(getattr(params, "AUGMENTATION", getattr(self.param.get("data_augmentation", {}), "ENABLE", True))),
            params_image_size=params.IMAGE_SIZE,
            params_classes=params.CLASSES,
            params_batch_size=params.BATCH_SIZE,
            params_include_top=params.INCLUDE_TOP,
            params_epochs=params.EPOCHS,
            params_weights=params.WEIGHTS,
            params_learning_rate=params.LEARNING_RATE,
            params_head_type=params.HEAD_TYPE,
            params_dropout_rate=float(params.DROPOUT_RATE),
            params_unfreeze_from_layer=str(params.UNFREEZE_FROM_LAYER) if params.UNFREEZE_FROM_LAYER is not None else "None",
            params_seed=int(getattr(params, "SEED", 42))
        )

        return baseline_model_config


    def get_training_config(self) -> TrainingConfig:

        training_config = self.config.training
        prepare_base_model_config = self.config.prepare_base_model
        params = self.param.baseline_model
        aug_params = self.param.data_augmentation
        cw_params = getattr(self.param, "class_weights", {})
        reduce_lr_params = getattr(self.param, "reduce_lr", {})

        create_directories([training_config.root_dir])

        training_config = TrainingConfig(
            root_dir=Path(training_config.root_dir),
            training_model_path=Path(training_config.training_model_path),
            updated_base_model_path=Path(prepare_base_model_config.updated_baseline_model_path),
            training_data=Path(self.config.data_preparation.prepared_dir),
            param_epochs=params.EPOCHS,
            params_patience=params.PATIENCE,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=bool(aug_params.ENABLE),
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_seed=int(getattr(params, "SEED", 42)),
            params_augmentation=dict(aug_params),
            params_class_weights=dict(cw_params),
            params_reduce_lr=dict(reduce_lr_params)
        )

        return training_config

    
    def get_evaluation_config(self) -> EvaluationConfig:
        combined_params = dict(self.param.baseline_model)
        if hasattr(self.param, "data_augmentation"):
            aug = self.param.data_augmentation
            combined_params["ROTATION_RANGE"] = aug.get("ROTATION_RANGE", 20)
            combined_params["ZOOM_RANGE"] = aug.get("ZOOM_RANGE", 0.2)
            combined_params["HORIZONTAL_FLIP"] = aug.get("HORIZONTAL_FLIP", True)
            combined_params["WIDTH_SHIFT_RANGE"] = aug.get("WIDTH_SHIFT_RANGE", 0.2)
            combined_params["HEIGHT_SHIFT_RANGE"] = aug.get("HEIGHT_SHIFT_RANGE", 0.2)
            combined_params["SHEAR_RANGE"] = aug.get("SHEAR_RANGE", 0.2)
            combined_params["BRIGHTNESS_RANGE"] = aug.get("BRIGHTNESS_RANGE", [0.8, 1.2])
            combined_params["FILL_MODE"] = aug.get("FILL_MODE", "nearest")
            combined_params["aug_enable"] = aug.get("ENABLE", True)
        if hasattr(self.param, "class_weights"):
            cw = self.param.class_weights
            combined_params["cw_enable"] = cw.get("ENABLE", False)
            combined_params["cw_mode"] = cw.get("MODE", "manual")
            if "CUSTOM_WEIGHTS" in cw:
                combined_params["CLASS_WEIGHTS"] = dict(cw.CUSTOM_WEIGHTS)
        if hasattr(self.param, "reduce_lr"):
            rlr = self.param.reduce_lr
            combined_params["reduce_lr_enable"] = rlr.get("ENABLE", False)
            combined_params["reduce_lr_factor"] = rlr.get("FACTOR", 0.5)
            combined_params["reduce_lr_patience"] = rlr.get("PATIENCE", 3)
            combined_params["reduce_lr_min_lr"] = rlr.get("MIN_LR", 1e-7)

        eval_config = EvaluationConfig(
            path_to_model=Path("artifacts/training/model.h5"),
            test_data=Path("artifacts/data_preparation/prepared/test"),
            mlflow_uri="https://dagshub.com/shlokbagwe.tech/Chest-Cancer-Classifier.mlflow",
            all_params=combined_params,
            params_image_size=self.param.baseline_model.IMAGE_SIZE,
            params_batch_size=self.param.baseline_model.BATCH_SIZE,
            best_model_dir=Path(getattr(self.config.training, "best_model_dir", "artifacts/training/best_model")),
        )
        return eval_config

