import os
import numpy as np
from PIL import Image as pil_image
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Initialize the Flask application
app = Flask(__name__)

# Load the lung cancer model
MODEL_PATH = "C:\\Users\\ashad\\Desktop\\praharshini_project\\lung_cancer_resnet_model.h5"
model = load_model(MODEL_PATH)

# Define the class labels
classes = [
    'Benign Lung Tumor',
    'Malignant Lung Tumor',
    'Normal Lung Tissue'
]

def model_predict(img_path, model):
    # Load the image and resize it to the required size
    img = image.load_img(img_path, target_size=(256, 256, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Preprocess the image
    x = tf.keras.applications.vgg16.preprocess_input(x)

    # Make the prediction
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process the result
        pred_class_idx = preds.argmax(axis=-1)[0]
        confidence = preds[0][pred_class_idx] * 100
        result = f"{classes[pred_class_idx]} with {confidence:.2f}% confidence"
        
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
