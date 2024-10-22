import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

# Load the newly trained model
model = tf.keras.models.load_model('face_40_transfer_learning.keras')

# Load class descriptions
with open('descriptions.json', 'r', encoding='utf-8') as f:
    class_descriptions = json.load(f)

# Define image size and class labels
img_height, img_width = 100, 100  # Match these with your new training size
class_labels = list(class_descriptions.keys())  # Get class labels from the JSON file

# Function to preprocess the uploaded image
def preprocess_image(uploaded_image):
    img = image.load_img(uploaded_image, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0  # Normalizing the image
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions and return the closest match
def predict_and_return_info(img_array):
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # Get class name from predicted index
    predicted_class_label = class_labels[predicted_class_index]
    
    # Reference image path (assuming you have reference images named as 'class1_image.jpg', etc.)
    ref_image_path = os.path.join('images', f"{predicted_class_label}_image.jpg")
    
    # If confidence is above a threshold (e.g., 0.92), return class info and reference image path
    if confidence > 0.8 and os.path.exists(ref_image_path):
        class_info = class_descriptions[predicted_class_label]
        return class_info['full_name'], class_info['description'], confidence, ref_image_path
    elif confidence > 0.8:
        class_info = class_descriptions[predicted_class_label]
        return class_info['full_name'], class_info['description'], confidence, None
    else:
        return None, None, confidence, None

# Custom CSS for styling the containers (container 1 with a background image and others with custom fonts/colors)
st.markdown("""
    <style>
    .container1 {
        background-color: #b06679;
        color: white;
        text-align: center;
        padding: 20px;
        font-size: 30px;
        border-radius: 10px;
    }
    .container2 {
        background-color: #cedead;
        padding: 20px;
        font-size: 18px;
        border-radius: 10px;
    }
    .container3 {
        background-color: #b08792;
        padding: 20px;
        font-size: 18px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Container 1 - Custom background image and styled text
with st.container():
    st.image("face_rec_image6.jpeg", use_column_width=True)

# Container 2 - File uploader and image recognition
with st.container():
    st.markdown('<div class="container1">Welcome to Group 2 EDSG X Gomycode Data Science Face Recognition App</div>', unsafe_allow_html=True)
    st.markdown('<div class="container2">Upload an image of a face and let\'s show you the magic!</div>', unsafe_allow_html=True)
    
    # File uploader
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(uploaded_image)

        # Get prediction and information
        class_name, description, confidence, ref_image_path = predict_and_return_info(img_array)

        # Display result
        if class_name:
            st.success(f"Predicted Class: {class_name}")
            st.info(f"Description: {description}")
            st.write(f"Confidence: {confidence * 100:.2f}%")

            # Display reference image if available
            if ref_image_path:
                st.image(ref_image_path, caption=f"Reference image for {class_name}", use_column_width=True)
            else:
                st.warning("No reference image available for this class.")
        else:
            st.error("No matching class found. Please upload a clearer image or try a different one.")

# Container 3 - Additional Information or Footer
with st.container():
    st.markdown('<div class="container3">Thank you for visiting our Face Recognition service!</div>', unsafe_allow_html=True)
