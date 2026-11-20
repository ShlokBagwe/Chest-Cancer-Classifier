import os
import re
import json
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import tensorflow as tf
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

        # 4. Check and promote to best_model/ if it beats champion
        self.archive_if_best()

    def archive_if_best(self):
        """
        Check if the newly evaluated model outperforms all previously archived models
        in best_model_dir. If it does, promote and archive it as model_v{N}.h5,
        save model_v{N}_metrics.json, and update champion_manifest.json.
        """
        best_model_dir = Path(self.config.best_model_dir)
        best_model_dir.mkdir(parents=True, exist_ok=True)

        current_loss = float(self.score[0])
        current_acc = float(self.score[1])

        # 1. Identify existing versioned models
        version_pattern = re.compile(r"^model_v(\d+)\.h5$")
        existing_versions = []
        for file_path in best_model_dir.glob("model_v*.h5"):
            match = version_pattern.match(file_path.name)
            if match:
                existing_versions.append(int(match.group(1)))

        # 2. Determine benchmark metrics from champion_manifest.json or individual metrics files
        manifest_path = best_model_dir / "champion_manifest.json"
        best_acc = -1.0
        best_loss = float("inf")
        champion_version = None
        history = []

        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest_data = json.load(f)
                best_acc = float(manifest_data.get("test_accuracy", -1.0))
                best_loss = float(manifest_data.get("test_loss", float("inf")))
                champion_name = manifest_data.get("current_champion", "")
                m = version_pattern.match(champion_name)
                if m:
                    champion_version = int(m.group(1))
                history = manifest_data.get("history", [])
            except Exception as e:
                logger.warning(f"Could not load {manifest_path}: {e}")

        # Fallback to inspecting individual model_v{N}_metrics.json if manifest was missing or incomplete
        if champion_version is None and existing_versions:
            for v in existing_versions:
                metrics_path = best_model_dir / f"model_v{v}_metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path, "r") as f:
                            m_data = json.load(f)
                        m_metrics = m_data.get("metrics", {})
                        v_acc = float(m_metrics.get("test_accuracy", m_metrics.get("accuracy", -1.0)))
                        v_loss = float(m_metrics.get("test_loss", m_metrics.get("loss", float("inf"))))
                        if v_acc > best_acc or (abs(v_acc - best_acc) < 1e-4 and v_loss < best_loss):
                            best_acc = v_acc
                            best_loss = v_loss
                            champion_version = v
                    except Exception as e:
                        logger.warning(f"Could not read {metrics_path}: {e}")

        # 3. Evaluate if current model beats benchmark
        is_better = False
        reason = ""

        if not existing_versions or champion_version is None:
            is_better = True
            reason = "Initial champion model archive"
        elif current_acc > best_acc + 1e-4:
            is_better = True
            reason = f"Test accuracy improved from {best_acc * 100:.2f}% (model_v{champion_version}) to {current_acc * 100:.2f}%"
        elif abs(current_acc - best_acc) <= 1e-4 and current_loss < best_loss - 1e-4:
            is_better = True
            reason = f"Test accuracy tied at {current_acc * 100:.2f}%, but test loss improved from {best_loss:.4f} to {current_loss:.4f}"
        else:
            is_better = False
            reason = (
                f"Current model (Acc: {current_acc * 100:.2f}%, Loss: {current_loss:.4f}) did not beat "
                f"champion model_v{champion_version} (Acc: {best_acc * 100:.2f}%, Loss: {best_loss:.4f})"
            )

        # 4. Promote and archive if champion
        if is_better:
            next_version = (max(existing_versions) + 1) if existing_versions else 1
            new_model_name = f"model_v{next_version}.h5"
            new_model_path = best_model_dir / new_model_name
            new_metrics_path = best_model_dir / f"model_v{next_version}_metrics.json"

            logger.info("=" * 60)
            logger.info("NEW CHAMPION MODEL DETECTED!")
            logger.info(f"Reason: {reason}")
            logger.info(f"Archiving champion as {new_model_name} in {best_model_dir}...")

            # Copy model weights
            shutil.copy2(self.config.path_to_model, new_model_path)

            # Compute detailed classification report
            report = classification_report(
                self.y_true,
                self.y_pred,
                target_names=self.class_names,
                output_dict=True
            )
            macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))
            weighted_f1 = float(report.get("weighted avg", {}).get("f1-score", 0.0))
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update previous champion's status
            if champion_version is not None:
                prev_metrics_file = best_model_dir / f"model_v{champion_version}_metrics.json"
                if prev_metrics_file.exists():
                    try:
                        with open(prev_metrics_file, "r") as f:
                            prev_data = json.load(f)
                        prev_data["status"] = f"Previous Champion (Superseded by {new_model_name})"
                        with open(prev_metrics_file, "w") as f:
                            json.dump(prev_data, f, indent=2)
                    except Exception as e:
                        logger.warning(f"Could not update previous champion status: {e}")

            # Save new champion metrics
            new_metadata = {
                "model_version": new_model_name,
                "archived_date": timestamp,
                "status": "Current Champion",
                "promotion_reason": reason,
                "metrics": {
                    "test_accuracy": current_acc,
                    "test_loss": current_loss,
                    "macro_f1": macro_f1,
                    "weighted_f1": weighted_f1,
                    "classification_report": report
                },
                "params": dict(self.config.all_params),
                "hyperparameters": dict(self.config.all_params),
                "supersedes": f"model_v{champion_version}.h5" if champion_version else None
            }
            save_json(path=new_metrics_path, data=new_metadata)

            # Update champion_manifest.json
            history_entry = {
                "version": new_model_name,
                "test_accuracy": current_acc,
                "test_loss": current_loss,
                "macro_f1": macro_f1,
                "date": timestamp,
                "status": "Current Champion",
                "reason": reason,
                "params": dict(self.config.all_params)
            }
            for h in history:
                if h.get("status") == "Current Champion":
                    h["status"] = f"Superseded by {new_model_name}"

            history.append(history_entry)

            manifest_content = {
                "current_champion": new_model_name,
                "champion_path": str(new_model_path),
                "test_accuracy": current_acc,
                "test_loss": current_loss,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "last_updated": timestamp,
                "champion_params": dict(self.config.all_params),
                "history": history
            }
            save_json(path=manifest_path, data=manifest_content)

            logger.info(f"Successfully archived {new_model_name} and updated manifest.")
            logger.info("=" * 60)
        else:
            logger.info("=" * 60)
            logger.info("Model did not qualify as new champion.")
            logger.info(f"Reason: {reason}")
            logger.info(f"Latest weights preserved in {self.config.path_to_model}; {best_model_dir} unchanged.")
            logger.info("=" * 60)

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
            mlflow_params = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in self.config.all_params.items()}
            mlflow.log_params(mlflow_params)
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
