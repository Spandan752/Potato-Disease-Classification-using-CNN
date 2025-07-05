import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import base64

st.set_page_config(page_title="Potato Disease Detection", page_icon=":seedling:", layout="wide")

st.title("Potato Disease Detection App")
st.write("Upload an image of a potato leaf to detect diseases.")

model = tf.keras.models.load_model("D:/Projects/Potato_Disease/models/CNN_model1.keras")
class_names = ["Early Blight", "Late Blight", "Healthy"]

uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width = 100, use_container_width=False)

    image_np = np.array(image)
    img_batch = np.expand_dims(image_np, 0)

    prediction = model.predict(img_batch)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    st.markdown(f"Predicted Class: {predicted_class}")
    st.markdown(f"Confidence: {confidence:.2f}")

