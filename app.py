# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Rice Disease Classifier", page_icon="🌾")

# Constants from your training
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Bacterial_leaf_blight', 'Brown_spot', 'Healthy', 'Leaf_blast']

# Cache the model loading
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('rice_disease_model.keras')

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Preprocessing function
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

# Streamlit interface
st.title("Rice Disease Classifier 🌾")
st.write("Upload an image of a rice leaf for disease diagnosis")

uploaded_file = st.file_uploader("Choose an image...", 
                               type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        # Read and display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed parameter here

        # Preprocess and predict
        with st.spinner('Analyzing...'):
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100

        # Display results
        st.subheader("Results")
        st.success(f"Predicted Disease: **{predicted_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # Show probability distribution
        st.subheader("Class Probabilities")
        for class_name, prob in zip(CLASS_NAMES, predictions[0]):
            st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
