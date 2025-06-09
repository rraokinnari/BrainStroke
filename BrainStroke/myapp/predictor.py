# predictor.py
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load model once
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Vgg-copy.h5')
model = load_model(MODEL_PATH)

# IMPORTANT: class_indices must match training generator's class_indices
# Set them manually as per your training generator
# For example, if test_gen.class_indices = {'Normal': 0, 'Stroke': 1}
class_indices = {'Normal': 0, 'Stroke': 1}
inv_class_indices = {v: k for k, v in class_indices.items()}

def predict_stroke(img_path, target_size=(224, 224)):
    try:
        # Load and resize image
        img = image.load_img(img_path, target_size=target_size)

        # Convert to array and expand dimensions
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # DO NOT normalize if your training didn’t use normalization
        # If your model expects unnormalized images, skip this:
        # img_array /= 255.0

        # Predict
        predictions = model.predict(img_array)

        # Get predicted index
        predicted_class_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class_index])

        # Get readable label
        predicted_label = inv_class_indices.get(predicted_class_index, f"Unknown Class ({predicted_class_index})")

        return predicted_label, confidence

    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0
    # Normalization function (if your model was trained on normalized images)
def scalar(img_array):
    return img_array / 255.0

