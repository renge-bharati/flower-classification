import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Flower Classification App",
    layout="centered"
)

st.title("🌸 Flower Classification App")
st.write("Upload a flower image to predict its type")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("pruned_flower_classifier_model.h5")
    return model

model = load_trained_model()

# Class labels (MUST match training order)
CLASS_NAMES = ["Lilly", "Lotus", "Orchid", "Sunflower", "Tulip"]

# -----------------------------
# Image Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((150, 150))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Result
    st.success(f"🌼 Predicted Flower: **{CLASS_NAMES[predicted_index]}**")
    st.info(f"Confidence: **{confidence:.2f}**")

