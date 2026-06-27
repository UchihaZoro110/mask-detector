import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image

model = joblib.load("final_model.pkl")
pca = joblib.load("final_pca.pkl")

st.title("Face Mask Detector")
st.write("Image upload karo — pata chalega mask hai ya nahi!")

uploaded_file = st.file_uploader("Image choose karo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    image_resized = cv2.resize(image, (150, 150))
    image_normalized = image_resized.astype('float32') / 255.0
    image_flat = image_normalized.reshape(1, -1)
    image_pca = pca.transform(image_flat)
    prediction = model.predict(image_pca)
    
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Uploaded Image", width=300)
    
    if prediction[0] == 1:
        st.success("✅ Mask pehna hua hai!")
    else:
        st.error("❌ Mask nahi pehna!")