import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
import mlflow
import mlflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

from ChestCancer.entity.config_entity import EvaluationConfig
from ChestCancer.utils.common import save_json
from ChestCancer import logger


# Prevent cuDNN allocation failures on limited VRAM
try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    logger.warning(f"Could not configure GPU memory growth: {e}")


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None

    def _valid_generator(self):
        """Prepare evaluation generator with VGG16 preprocess_input."""
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        self.valid_generator = val_datagen.flow_from_directory(
            directory=self.config.test_data,
            target_size=tuple(self.config.params_image_size[:-1]),  # (224, 224)
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
            shuffle=False,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(str(path))

    def evaluation(self):
        """Load trained model and evaluate on test generator."""
        
        self.model = self.load_model(self.config.path_to_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        logger.info(f"Evaluation finished: Loss = {self.score[0]:.4f}, Accuracy = {self.score[1]:.4f}")
        self.save_score()



    def save_score(self):
        """Save test loss and accuracy to scores.json."""
        
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        save_json(path=Path("scores.json"), data=scores)


    def log_into_mlflow(self):
        """Log parameters, metrics, and trained model into MLflow / DagsHub."""
        mlflow.set_registry_uri(self.config.mlflow_uri)
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            # Model registry does not work with file store (local)
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
