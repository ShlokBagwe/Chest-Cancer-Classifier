from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Linux/WSL
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

from ChestCancer import logger
from ChestCancer.entity.config_entity import TrainingConfig

# Prevent cuDNN allocation failures on limited VRAM
try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    logger.warning(f"Could not configure GPU memory growth: {e}")


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None


    def _get_base_model(self):
        """Loads the updated base model and re-compiles it with the configured learning rate."""
        
        logger.info(f"Loading updated base model from: {self.config.updated_base_model_path}")

        self.model = keras.models.load_model(str(self.config.updated_base_model_path))



    def _train_valid_generator(self):
        """
         train/valid generators with VGG16 preprocess_input and optional augmentation
         """
        
        flow = dict(
            target_size=tuple(self.config.params_image_size[:-1]),  # drop channel: (224, 224)
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
        )

        if self.config.params_is_augmentation:
            logger.info("Configuring training generator with augmentation and preprocess_input")

            aug = dict(
                rotation_range=20,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,

            )

        else:
            logger.info("Configuring training generator with preprocess_input (augmentation disabled)")
            aug = {}


        # Note: not adding rescale=1./255 because VGG16 preprocess_input already performs it
        
        train_gen = ImageDataGenerator(preprocessing_function=preprocess_input, **aug)
        valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_dir = self.config.training_data / "train"
        valid_dir = self.config.training_data / "valid"

        logger.info(f"Loading training data from: {train_dir}")
        self.train_generator = train_gen.flow_from_directory(Path(train_dir), shuffle=True, **flow)


        logger.info(f"Loading validation data from: {valid_dir}")
        self.valid_generator = valid_gen.flow_from_directory(Path(valid_dir), shuffle=False, **flow)


    @staticmethod
    def _save_model(path: Path, model: tf.keras.Model):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        logger.info(f"Model saved successfully at: {path}")


    def train(self, callback_list: list = None):
        
        if self.model is None:
            self._get_base_model()

        if self.train_generator is None or self.valid_generator is None:
            self._train_valid_generator()
        

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        logger.info(f"Model compiled with learning rate: {self.config.params_learning_rate}")



        if callback_list is None:
            callback_list = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.params_patience,
                    restore_best_weights=True,
                    verbose=1,
                )
            ]

        logger.info(f"Starting training for up to {self.config.param_epochs} epochs...")

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.param_epochs,
            validation_data=self.valid_generator,
            callbacks=callback_list,
        )

        history_path = Path(self.config.training_model_path).parent / "history.json"
        with open(history_path, "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=4)


        logger.info("Training finished")
        
        self._save_model(self.config.training_model_path, self.model)

        self._save_plot(history)

        return history

    
    def _save_plot(self, history):
        """Saves train vs validation loss and accuracy curves as an image."""
        plot_path = Path(self.config.training_model_path).parent / "training_curves.png"
        plt.figure(figsize=(12, 5))

        # 1. Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss", color="royalblue", linewidth=2)
        plt.plot(history.history["val_loss"], label="Validation Loss", color="darkorange", linewidth=2)
        plt.title("Loss vs. Epochs", fontsize=12)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # 2. Accuracy Curve
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Train Accuracy", color="royalblue", linewidth=2)
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="darkorange", linewidth=2)
        plt.title("Accuracy vs. Epochs", fontsize=12)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logger.info(f"Training curves saved at: {plot_path}")