import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model (Ensure 'flowermodel01.h5' is in the same directory)
model = load_model('flowermodel01.h5')

# Class names (Adjust according to your dataset's class names)
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def preprocess_image(image):
    img = image.resize((150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return class_names[np.argmax(prediction)], np.max(prediction)

# Streamlit interface
st.title("Flower Classification Web App")
st.write("Upload an image of a flower, and the model will classify it.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")
    label, confidence = predict(image)
    st.write(f"Prediction: {label} ({confidence*100:.2f}%)")