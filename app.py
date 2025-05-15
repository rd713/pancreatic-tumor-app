import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown  # install with: pip install gdown

# Constants
MODEL_PATH = "best_aunet_model (1).keras"
# Replace with your actual Google Drive "shareable link" ID for the model file
# Example link: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
MODEL_URL = "https://drive.google.com/file/d/19SQhrtegE5i3u2Y0ZtvArhd6miP9wuGU/view?usp=sharing"

# Download model if not present
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model, please wait...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return MODEL_PATH

# Load model (after download)
@st.cache_resource
def load_model():
    model_file = download_model()
    model = tf.keras.models.load_model(model_file, compile=False)
    return model

model = load_model()

# Preprocessing function (adjust size & grayscale channel)
def preprocess_image(image: Image.Image, target_size=(128, 128)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    if image_array.ndim == 2:  # Grayscale image
        image_array = np.expand_dims(image_array, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA -> RGB
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)  # Batch dimension
    return image_array

# Postprocess prediction mask
def postprocess_mask(pred, original_size):
    pred_mask = pred[0, :, :, 0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask = Image.fromarray(pred_mask).resize(original_size)
    return pred_mask

# Streamlit UI
st.title("Pancreatic Tumor Segmentation")

uploaded_file = st.file_uploader("Upload a CT Scan Image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Segment Tumor"):
        with st.spinner("Segmenting..."):
            preprocessed = preprocess_image(image)
            prediction = model.predict(preprocessed)
            result_mask = postprocess_mask(prediction, image.size)
            st.image(result_mask, caption="Tumor Segmentation Mask", use_column_width=True)
