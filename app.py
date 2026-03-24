import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("Fake-currency.keras")

# Function to classify currency with confidence score
def classify_currency(image):
    img = image.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for model input

    prediction = model.predict(img_array)[0][0]  # Get prediction score
    confidence = round(prediction * 100, 2)  # Convert to percentage

    if prediction > 0.5:
        return f"🟢 Real Currency (Confidence: {confidence}%)", confidence
    else:
        return f"🔴 Fake Currency (Confidence: {100 - confidence}%)", confidence

# Function to detect watermark using edge detection
def detect_watermark(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)  # Detect edges
    return edges

# Function to detect texture differences
def analyze_texture(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()  # Measures sharpness/texture clarity
    return laplacian

# Streamlit Web App
st.title("💵 Fake Currency Detector")
st.write("Upload currency images for analysis.")

# Allow two file uploads
image_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if image_files and len(image_files) == 2:
    image1 = Image.open(image_files[0])
    image2 = Image.open(image_files[1])

    # Process first image
    result1, confidence1 = classify_currency(image1)
    watermark1 = detect_watermark(image1)
    texture_value1 = analyze_texture(image1)

    # Process second image
    result2, confidence2 = classify_currency(image2)
    watermark2 = detect_watermark(image2)
    texture_value2 = analyze_texture(image2)

    # Display results above the first set of images
    st.markdown(f"### 📝 Currency 1: {result1}")
    st.write(f"🧐 **Texture Clarity Score:** {round(texture_value1, 2)}")
    if texture_value1 < 50:
        st.warning("⚠️ Low texture clarity detected – Possible Fake Currency!")
    
    # Display first set of images
    cols = st.columns(2)
    with cols[0]:
        st.image(image1, caption="📸 Currency Note 1", use_container_width=True)
    with cols[1]:
        st.image(watermark1, caption="🖼 Watermark Detection", use_container_width=True, channels="GRAY")
    
    # Display second set of images
    cols = st.columns(2)
    with cols[0]:
        st.image(image2, caption="📸 Currency Note 2", use_container_width=True)
    with cols[1]:
        st.image(watermark2, caption="🖼 Watermark Detection", use_container_width=True, channels="GRAY")
    
    # Display results below the second set of images
    st.markdown(f"### 📝 Currency 2: {result2}")
    st.write(f"🧐 **Texture Clarity Score:** {round(texture_value2, 2)}")
    if texture_value2 < 50:
        st.warning("⚠️ Low texture clarity detected – Possible Fake Currency!")
