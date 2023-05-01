# Predicting a new image using our CNN model
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import cv2
import os
from mtcnn_cv2 import MTCNN
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import streamlit as st


model = tf.keras.models.load_model(r"dmg_car-weights-CNN.h5")

#

def predict_image(file_path, model, threshold):
    # Load and preprocess the image
    img = image.load_img(file_path, target_size=model.input_shape[1:4])
    img_array = image.img_to_array(img)
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

  uploaded_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])
  threshold=0.7
  if uploaded_file is not None:
     path_in = uploaded_file.name
     #file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
     #image = cv2.imdecode(file_bytes, 1)
     # Call the function and display the result in Streamlit
     result=predict_image(path_in, model,threshold)
     #st.image(image, caption='Input image', use_column_width=True)
     st.write('Prediction:', result)
     
     
     

if __name__ == "__main__":
    main()
