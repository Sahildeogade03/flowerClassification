import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model (Ensure 'flowermodel01.h5' is in the same directory)
@st.cache_resource
def load_trained_model():
    return load_model('best_model.h5')

model = load_trained_model()

# Class names (Ensure they match your dataset)
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Function to preprocess image (Fixes shape issue)
def preprocess_image(image):
    try:
        img = image.convert("RGB")  # Ensure RGB format
        img = img.resize((256, 256))  # ✅ Resize to match model input (256x256)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize
        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Function to predict flower type
def predict(image):
    processed_img = preprocess_image(image)
    if processed_img is None:
        return None
    
    predictions = model.predict(processed_img)[0]  # Get prediction scores
    top_3_indices = np.argsort(predictions)[-3:][::-1]  # Get top 3 predictions
    top_3_labels = [(class_names[i], predictions[i]) for i in top_3_indices]
    return top_3_labels

# Streamlit App UI
st.title("🌸 Flower Classification App")
st.write("Upload an image of a flower, and the model will classify it into one of the five categories.")

# Upload Image
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)
    
    st.write("🔍 **Classifying...**")
    
    top_3_predictions = predict(image)

    if top_3_predictions is None:
        st.error("⚠️ Unable to classify image. Please try again with a different image.")
    else:
        st.subheader("🧠 Prediction Results:")
        for label, confidence in top_3_predictions:
            st.write(f"**{label.capitalize()}** - {confidence * 100:.2f}%")
