import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import dagshub
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
        self.y_true = None
        self.y_pred = None
        self.class_names = None

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
        """Load trained model, evaluate loss/acc, generate predictions and confusion matrix."""
        self.model = self.load_model(self.config.path_to_model)
        self._valid_generator()
        
        # 1. Evaluate loss and accuracy
        self.score = self.model.evaluate(self.valid_generator)
        logger.info(f"Evaluation finished: Loss = {self.score[0]:.4f}, Accuracy = {self.score[1]:.4f}")

        # 2. Get predictions and true labels
        predictions = self.model.predict(self.valid_generator)
        self.y_pred = np.argmax(predictions, axis=1)
        self.y_true = self.valid_generator.classes
        self.class_names = list(self.valid_generator.class_indices.keys())

        # 3. Save scores and confusion matrix
        self.save_score()
        self.save_confusion_matrix()

    def save_score(self):
        """Save test loss, accuracy, and classification report to JSON."""
        report = classification_report(
            self.y_true,
            self.y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        scores = {
            "loss": float(self.score[0]),
            "accuracy": float(self.score[1]),
            "classification_report": report
        }
        save_json(path=Path("scores.json"), data=scores)

        eval_dir = Path("artifacts/evaluation")
        eval_dir.mkdir(parents=True, exist_ok=True)
        save_json(path=eval_dir / "classification_report.json", data=report)
        logger.info("Scores and classification report saved.")

    def save_confusion_matrix(self):
        """Plot and save ConfusionMatrixDisplay image."""
        eval_dir = Path("artifacts/evaluation")
        eval_dir.mkdir(parents=True, exist_ok=True)
        cm_path = eval_dir / "confusion_matrix.png"

        cm = confusion_matrix(self.y_true, self.y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap="Blues", xticks_rotation=45, ax=ax)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved at: {cm_path}")

    def log_into_mlflow(self):
        """Log parameters, metrics, artifacts, and trained model into MLflow / DagsHub."""
        dagshub.init(repo_owner="shlokbagwe.tech", repo_name="Chest-Cancer-Classifier", mlflow=True)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            # Log artifacts: confusion matrix image and classification report
            cm_path = Path("artifacts/evaluation/confusion_matrix.png")
            report_path = Path("artifacts/evaluation/classification_report.json")
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))
            if report_path.exists():
                mlflow.log_artifact(str(report_path))

            # Model registry does not work with file store (local)
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
