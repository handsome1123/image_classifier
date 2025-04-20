# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 13:50:35 2025

@author: LAB
"""

import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle
import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Set title
st.title("Image Classification with MobileNetV2 by Saw San Nyunt Win")

# File upload
upload_file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    # Display uploaded image
    img = Image.open(upload_file)
    st.image(img, caption="Uploaded Image")  # <-- FIXED LINE

    # Preprocessing
    img = img.resize((224, 224))  # ensure size is a tuple
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Prediction
    preds = model.predict(x)
    top_preds = decode_predictions(preds, top=3)[0]

    # Display predictions
    st.subheader("Prediction:")
    for i, pred in enumerate(top_preds):
        st.write(f"{i+1}. **{pred[1]}** - {round(pred[2]*100, 2)}%")
