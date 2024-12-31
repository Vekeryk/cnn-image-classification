import os
import cv2
import numpy as np
import urllib.request
from flask import Flask, request, jsonify, render_template
from keras.api.models import load_model

app = Flask(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = load_model('best_model.keras')

class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def preprocess_image(image):
    # convert image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize image to the input size expected by the model
    image = cv2.resize(image, (32, 32))
    # normalize the image
    mean, std = np.mean(image), np.std(image)
    image = (image - mean) / (std + 1e-7)
    # add batch dimension
    image = image.reshape((1, 32, 32, 3))
    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' in request.files:
            file = request.files['image']
            image = np.asarray(bytearray(file.read()), dtype="uint8")
            if image.size == 0:
                return jsonify({'error': 'Invalid image file.'}), 400

            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        elif 'url' in request.json:
            url = request.json['url']
            resp = urllib.request.urlopen(url)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        else:
            return jsonify({'error': 'No image file or URL provided.'}), 400

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        predicted_class = prediction.argmax()

        return jsonify({
            'predicted_class': class_names[predicted_class],
            'confidence': round(float(prediction[0][predicted_class]), 8)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
