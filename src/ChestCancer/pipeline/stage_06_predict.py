import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input


class PredictPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # 1. Load trained model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        # 2. Load and preprocess image
        image_name = self.filename
        test_image = image.load_img(image_name, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = preprocess_input(test_image)

        # 3. Predict class
        predictions = model.predict(test_image)
        result = np.argmax(predictions, axis=1)

        # 4. Map index to class label
        classes = ["adenocarcinoma", "large_cell", "normal", "squamous_cell"]
        prediction = classes[result[0]]

        # 5. Return JSON-compatible list
        return [{"image": prediction}]
