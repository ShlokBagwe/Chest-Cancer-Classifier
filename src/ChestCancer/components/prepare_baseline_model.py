from ChestCancer.entity.config_entity import PrepareBaselineModelConfig
from pathlib import Path
 
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
 
from ChestCancer import logger


class PrepareBaselineModel:
    def __init__(self, config: PrepareBaselineModelConfig):
        self.config = config

    def get_baseline_model(self):
        """Load VGG16 (ImageNet weights, no top) and save it as the raw base model."""
        self.model = VGG16(
            weights=self.config.params_weights,
            include_top=self.config.params_include_top,
            input_shape=tuple(self.config.params_image_size)
        )

        PrepareBaselineModel.save_model(self.config.baseline_model_path, self.model)

    @staticmethod
    def _prepare_full_model(model: tf.keras.Model, classes: int, learning_rate: float):
        """Freeze the backbone, add a classification head and compile"""
        model.trainable = False
        x = Flatten()(model.output)
        prediction = Dense(classes, activation="softmax")(x)
        full_model = Model(model.input, outputs = prediction)
        
        full_model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss = 'categorical_crossentropy',
            metrics=['accuracy']
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        """Building full model on top of the base model"""
    
        self.full_model = self._prepare_full_model(self.model, self.config.params_classes, learning_rate=self.config.params_learning_rate) # can also be written as PrepareBaselineModel._prepare_full_model(...)

        self.save_model(path=self.config.updated_baseline_model_path, model=self.full_model)
        logger.info(f"Updated base model saved at {self.config.updated_baseline_model_path}")


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
