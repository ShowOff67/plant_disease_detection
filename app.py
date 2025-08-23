from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import io # Import the io library for in-memory processing
import json

app = Flask(__name__)

# ----------------------------
# Load trained model once
# ----------------------------
# This will happen when the app is first started on Render.
try:
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class labels
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ----------------------------
# Prediction function
# ----------------------------
def model_prediction(img_data):
    if not model:
        return -1 # Return an error index if the model is not loaded

    # Use io.BytesIO to read image data from memory
    image = tf.keras.preprocessing.image.load_img(io.BytesIO(img_data), target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    try:
        with open("training_hist.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = {} # Handle case where training_hist.json might not exist
    return render_template("home.html", history=history)

@app.route("/disease")
def disease():
    return render_template("disease.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img_data = file.read() # Read file data into memory
        result_index = model_prediction(img_data)
        
        if result_index == -1:
            return jsonify({"error": "Model could not be loaded."}), 500

        prediction = class_name[result_index]
        return jsonify({"prediction": prediction})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500 # Return a 500 error on any other exception

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)