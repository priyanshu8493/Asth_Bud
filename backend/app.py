from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS

# Load your trained model (use the .keras or .h5 file as you prefer)
model = tf.keras.models.load_model("model/aqi_model_v1.keras")

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features")

        if not features or len(features) != 24 or any(len(row) != 12 for row in features):
            return jsonify({"error": "Input must be a 24x12 list of floats"}), 400

        # Convert to numpy array and reshape to match model's expected input
        input_features = np.array(features, dtype=np.float32).reshape(1, 24, 12)

        prediction = model.predict(input_features)
        result = prediction[0][0]  # Assuming a single output value

        return jsonify({"prediction": float(result)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
