import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model('BatikDetection.h5')

# Dictionary for labels
dic = {0 : 'Betawi',
       1 : 'Kawung',
       2 : 'Megamendung',
       3 : 'Parang',
       4 : 'Sekar Jagad'}

# Prediction function
def predict_label(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)
    pred = np.argmax(model.predict(img), axis=-1)
    return dic[pred[0]]

# Streamlit UI
st.title("Batik Motif Classification")
st.write("Upload an image of batik to classify its motif")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    st.write("")
    st.write("Classifying...")
    label = predict_label(img)
    st.write(f"The image is classified as: **{label}**")
