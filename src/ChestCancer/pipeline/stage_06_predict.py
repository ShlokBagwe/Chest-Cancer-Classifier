from __unknown__ import class_names
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

class PredictPipeline:

    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = load_model(os.path.join("model","model.h5"))
        image_name = self.filename
        test_image = image.load_img(image_name, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = preprocess_input(test_image)

        predictions = model.predict(test_image)
        result = np.argmax(predictions, axis=1)

        classes = ["adenocarcinoma", "large_cell", "normal", "squamous_cell"]

        prediction = class_names[result[0]]

        # return in JSON style format
        return [{
            "image": prediction
        }]


