import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# 1. Center crop and resize as per training
def crop_center(img, cropx=200, cropy=200):
    y, x, _ = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:starty + cropy, startx:startx + cropx]

def scalar(img):
    img = crop_center(img, cropx=200, cropy=200)
    img = tf.image.resize(img, (224, 224)).numpy()
    return img

# 2. Load image for Grad-CAM
def load_img_for_gradcam(img_path):
    img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = scalar(img_array)  # Apply cropping + resizing
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, np.array(img) / 255.0  # Normalized original for overlay

# 3. Get Grad-CAM heatmap
def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]
    
    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 4. Generate Grad-CAM Overlay
def generate_gradcam_overlay(img_path, model, class_indices, last_conv_layer_name='block5_conv3'):
    img_array, raw_img = load_img_for_gradcam(img_path)
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)

    # Resize to match input
    heatmap = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on image
    superimposed_img = heatmap_color * 0.4 + raw_img * 255
    superimposed_img = np.uint8(superimposed_img)

    # Predict class and confidence
    preds = model.predict(img_array)
    class_labels = {v: k for k, v in class_indices.items()}
    pred_class = class_labels[np.argmax(preds)]
    confidence = 100 * np.max(preds)

    return superimposed_img, pred_class, confidence
