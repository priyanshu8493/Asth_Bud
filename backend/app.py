# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load your trained model (adjust file name as needed)
model = tf.keras.models.load_model('model/aqi_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0][0]
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
