"""
Flask application for deploying the PPG Heart Rate Prediction Model.
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model at startup to avoid reloading per request
model = tf.keras.models.load_model('hr_model.keras')

@app.route("/")
def home():
    """Renders the main frontend page."""
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    """
    Endpoint to predict heart rate from a given PPG signal window.
    Expects a JSON payload with a 'ppg_signal' array.
    """
    try:
        data = request.json.get("ppg_signal", [])
        if not data:
            return jsonify({"error": "No PPG signal data provided."}), 400

        if len(data) != 1000:
            return jsonify({"error": f"Invalid signal length. Expected exactly 1000 sequential readings (8s at 125Hz), got {len(data)}."}), 400

        signal = np.array(data, dtype=np.float32)
        
        # Standardize the signal
        std_dev = signal.std()
        if std_dev == 0:
            return jsonify({"error": "Invalid signal with zero standard deviation."}), 400
            
        signal = (signal - signal.mean()) / std_dev
        
        # Reshape to match the model's expected 3D input: (Batch Size, Sequence Length, Channels)
        # Using len(signal) avoids hardcoding the window length
        signal = signal.reshape(1, len(signal), 1)

        # Predict heart rate (returns an array of shape (1, 1), so extract scalar)
        hr_prediction = model.predict(signal)[0][0]

        return jsonify({"heart_rate": round(float(hr_prediction), 1)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
