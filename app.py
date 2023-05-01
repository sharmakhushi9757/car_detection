# Predicting a new image using our CNN model
import numpy as np
#from keras.preprocessing import image
from keras.utils import load_img, img_to_array
import tensorflow as tf
import cv2
import os
from mtcnn_cv2 import MTCNN
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st


model = tf.keras.models.load_model(r"dmg_car-weights-CNN.h5")

#
def predict_image(img, model, threshold):
    # Load and preprocess the image
    img = load_img(img, target_size=model.input_shape[1:4])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions using the model
    prediction = model.predict(img_array)[0][0]
    
    # Check if the prediction is above the threshold
    if prediction >= threshold:
        result = 'good'
    else:
        result = 'damaged'
    
    # Return the result
    return result




#

def main():
  st.title("Car Detection")
  threshold=0.7
  uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
  if uploaded_file is not None:
        st.image(uploaded_file, caption='Input image', use_column_width=True)
        if st.button('Predict'):
            result = predict_image(uploaded_file, model, threshold)
            st.write('Prediction:', result)

if __name__ == "__main__":
    main()
