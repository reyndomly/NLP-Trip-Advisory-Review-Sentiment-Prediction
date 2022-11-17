import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from preprocess import clean_text, transform
from sklearn.preprocessing import LabelBinarizer
from keras.utils import pad_sequences

app = Flask(__name__)

# Load the model
model = load_model('model.h5')
label = pickle.load(open('label.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    clean = clean_text(data)
    trans = transform(clean)
    padded = pad_sequences(trans, maxlen=500, padding='post')
    pred = model.predict(padded)
    result = label.inverse_transform(pred)[0]
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=8080, use_reloader=False)
