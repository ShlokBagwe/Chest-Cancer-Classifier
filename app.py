import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

from ChestCancer.utils.common import decodeImage
from ChestCancer.pipeline.stage_06_predict import PredictPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictPipeline(self.filename)


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    return "Training executed successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    if request.json and 'image' in request.json:
        image = request.json['image']
        decodeImage(image, clApp.filename)
    elif 'file' in request.files:
        file = request.files['file']
        file.save(clApp.filename)
    elif 'image' in request.files:
        file = request.files['image']
        file.save(clApp.filename)
    else:
        return jsonify({"error": "No image provided"}), 400

    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host='0.0.0.0', port=8080)

