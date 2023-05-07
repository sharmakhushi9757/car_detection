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
model_location= tf.keras.models.load_model(r"models/model3_loc.h5")
model_severity= tf.keras.models.load_model(r"models/model3_sev.h5")

#


def predictimage(img, model, threshold):
    # Load and preprocess the image
    img = load_img(img, target_size=model.input_shape[1:4])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions using the model
    prediction = model.predict(img_array)
    
    # Return prediction and explanation as a dictionary
    return prediction


def predictimage_1(img, model, threshold):
    # Load and preprocess the image
    img = load_img(img, target_size=model.input_shape[1:4])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions using the model
    prediction = model.predict(img_array)
    
    # Return prediction and explanation as a dictionary
    return prediction

def predict_image(img, model, threshold):
    # Load and preprocess the image
    img = load_img(img, target_size=model.input_shape[1:4])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions using the model
    prediction = model.predict(img_array)[0][0]
    
    # Return prediction and explanation as a dictionary
    return prediction


#

st.title('Car Damage Predictor')
st.write("---")
st.markdown('Upload an image of a car to see if it is damaged or not.')
st.sidebar.title('Settings')
threshold = st.sidebar.slider('Threshold', 0.0, 1.0, 0.5, 0.01)
def main():
  uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
  if st.button('Predict'):
        progress_bar = st.progress(0)
        result = predict_image(uploaded_file, model, threshold)
         # Check if the prediction is above the threshold
        if result >= 0.5:
            st.write("Are you sure that your car is damaged? Please submit another picture of the damage.")
            st.write("Hint: Try zooming in/out, using a different angle or different lighting")    
        else:
            st.write('damaged')
            st.write('Validation complete - proceed to location and severity determination')
            result1=predictimage(uploaded_file, model_location, threshold)
            pred_labels = np.argmax(result1, axis=1)
            d = {0:'Front', 1:'Rear', 2:'Side'}
            for key in d.keys():
                if pred_labels[0] == key:
                    st.write("Validating location of damage....Result:",d[key])
            result2=predictimage_1(uploaded_file, model_location, threshold)
            pred_labels_1 = np.argmax(result2, axis=1)
            d_1 = {0:'minor', 1:'moderate', 2:'severe'}
            for key in d_1.keys():
                if pred_labels_1[0] == key:
                    st.write("Validating severity of damage....Result:",d_1[key])
            st.write("Severity assessment complete.")
        progress_bar.progress(100)
  else:
    st.warning('Please upload an image.')
        


main()
