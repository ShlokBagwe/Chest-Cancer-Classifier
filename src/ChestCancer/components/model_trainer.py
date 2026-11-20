from pathlib import Path
from datetime import datetime
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Linux/WSL
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

from ChestCancer import logger
from ChestCancer.utils.common import set_seed
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
        set_seed(self.config.params_seed)
        
        flow = dict(
            target_size=tuple(self.config.params_image_size[:-1]),  # drop the channel: (224, 224)
            batch_size=self.config.params_batch_size,
            class_mode="categorical",
        )

        if self.config.params_is_augmentation:
            logger.info("Configuring training generator with augmentation and preprocess_input")

            aug_cfg = self.config.params_augmentation or {}
            aug = dict(
                rotation_range=aug_cfg.get("ROTATION_RANGE", 20),
                horizontal_flip=aug_cfg.get("HORIZONTAL_FLIP", True),
                width_shift_range=aug_cfg.get("WIDTH_SHIFT_RANGE", 0.2),
                height_shift_range=aug_cfg.get("HEIGHT_SHIFT_RANGE", 0.2),
                shear_range=aug_cfg.get("SHEAR_RANGE", 0.2),
                zoom_range=aug_cfg.get("ZOOM_RANGE", 0.2),
                brightness_range=list(aug_cfg.get("BRIGHTNESS_RANGE", [0.8, 1.2])),
                fill_mode=aug_cfg.get("FILL_MODE", "nearest"),
            )
            logger.info(f"Augmentation parameters: {aug}")

        else:
            logger.info("Configuring training generator with preprocess_input (augmentation disabled)")
            aug = {}


        # Note: not adding rescale=1./255 because VGG16 preprocess_input already performs it
        
        train_gen = ImageDataGenerator(preprocessing_function=preprocess_input, **aug)
        valid_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_dir = self.config.training_data / "train"
        valid_dir = self.config.training_data / "valid"

        logger.info(f"Loading training data from: {train_dir}")
        self.train_generator = train_gen.flow_from_directory(
            Path(train_dir),
            shuffle=True,
            seed=self.config.params_seed,
            **flow
        )


        logger.info(f"Loading validation data from: {valid_dir}")
        self.valid_generator = valid_gen.flow_from_directory(
            Path(valid_dir),
            shuffle=False,
            seed=self.config.params_seed,
            **flow
        )


    @staticmethod
    def _save_model(path: Path, model: tf.keras.Model):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        logger.info(f"Model saved successfully at: {path}")


    def train(self, callback_list: list = None):
        set_seed(self.config.params_seed)
        
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

            reduce_lr_cfg = getattr(self.config, "params_reduce_lr", {})
            if reduce_lr_cfg and reduce_lr_cfg.get("ENABLE", False):
                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=float(reduce_lr_cfg.get("FACTOR", 0.5)),
                    patience=int(reduce_lr_cfg.get("PATIENCE", 3)),
                    min_lr=float(reduce_lr_cfg.get("MIN_LR", 1e-7)),
                    min_delta=float(reduce_lr_cfg.get("MIN_DELTA", 0.001)),
                    verbose=int(reduce_lr_cfg.get("VERBOSE", 1)),
                )
                callback_list.append(lr_scheduler)
                logger.info(
                    f"Configured ReduceLROnPlateau callback: factor={reduce_lr_cfg.get('FACTOR', 0.5)}, "
                    f"patience={reduce_lr_cfg.get('PATIENCE', 3)}, min_lr={reduce_lr_cfg.get('MIN_LR', 1e-7)}"
                )

        # Resolve class weights if enabled
        class_weight_dict = None
        cw_cfg = self.config.params_class_weights
        if cw_cfg and cw_cfg.get("ENABLE", False):
            mode = str(cw_cfg.get("MODE", "balanced")).lower()
            if mode == "balanced":
                import numpy as np
                classes, counts = np.unique(self.train_generator.classes, return_counts=True)
                n_samples = len(self.train_generator.classes)
                n_classes = len(classes)
                class_weight_dict = {
                    int(cls): float(n_samples / (n_classes * count))
                    for cls, count in zip(classes, counts)
                }
                idx_to_class = {v: k for k, v in self.train_generator.class_indices.items()}
                readable_cw = {idx_to_class.get(k, k): round(w, 4) for k, w in class_weight_dict.items()}
                logger.info(f"Computed balanced class weights: {readable_cw} (indices: {class_weight_dict})")
            elif mode == "manual":
                custom = cw_cfg.get("CUSTOM_WEIGHTS", {})
                class_weight_dict = {
                    self.train_generator.class_indices[k]: float(v)
                    for k, v in custom.items()
                    if k in self.train_generator.class_indices
                }
                logger.info(f"Loaded manual class weights: {class_weight_dict}")
            else:
                logger.warning(f"Unknown class_weights MODE '{mode}'. Running without class weights.")
        else:
            logger.info("Class weights disabled (uniform 1.0 weighting)")

        logger.info(f"Starting training for up to {self.config.param_epochs} epochs...")

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.param_epochs,
            validation_data=self.valid_generator,
            callbacks=callback_list,
            class_weight=class_weight_dict
        )

        history_path = Path(self.config.training_model_path).parent / "history.json"
        with open(history_path, "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=4)


        logger.info("Training finished")
        
        self._save_model(self.config.training_model_path, self.model)

        self._save_plot(history)

        return history

    
    def _save_plot(self, history):
        """Saves train vs validation loss and accuracy curves as an image with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_dir = Path(self.config.training_model_path).parent
        plot_path = parent_dir / f"training_curves_{timestamp}.png"
        latest_plot_path = parent_dir / "training_curves.png"

        plt.figure(figsize=(12, 5))

        # 1. Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Train Loss", color="royalblue", linewidth=2)
        plt.plot(history.history["val_loss"], label="Validation Loss", color="darkorange", linewidth=2)
        plt.title(f"Loss vs. Epochs ({timestamp})", fontsize=12)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # 2. Accuracy Curve
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Train Accuracy", color="royalblue", linewidth=2)
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="darkorange", linewidth=2)
        plt.title(f"Accuracy vs. Epochs ({timestamp})", fontsize=12)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.savefig(latest_plot_path, dpi=300)
        plt.close()
        logger.info(f"Training curves saved at: {plot_path} and {latest_plot_path}")