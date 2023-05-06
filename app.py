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


model_damage = tf.keras.models.load_model(r"damagemodel.h5")
model_location= tf.keras.models.load_model(r"locationmodel.h5")
model_serverity= tf.keras.models.load_model(r"serverity_model.h5")
#
def predict_image(img, model, threshold):
    # Load and preprocess the image
    img = load_img(img, target_size=model.input_shape[1:4])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions using the model
    prediction = model.predict(img_array)
    return prediction

def pipe31(image_path, model):
    urllib.request.urlretrieve(image_path, 'save.jpg')
    img = load_img('save.jpg', target_size=(256,256))
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)/255
    pred = model.predict(x)
    pred_labels = np.argmax(pred, axis=1)
    d = {0:'Front', 1:'Rear', 2:'Side'}
    for key in d.keys():
        if pred_labels[0] == key:
            print("Validating location of damage....Result:",d[key])
    print("Severity assessment complete.")

def predictimage(img, model, threshold):
    # Load and preprocess the image
    img = load_img(img, target_size=model.input_shape[1:4])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make predictions using the model
    prediction = model.predict(img_array)
    return prediction
    
    
    # Return prediction and explanation as a dictionary


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
        result=predict_image(uploaded_file, model_damage, threshold)
        progress_bar.progress(100)
        if(result[0][0]>=0.5):
            st.write("Validation complete - proceed to location and severity determination")
            result1=predictimage(uploaded_file, model_location, threshold)
            pred_labels = np.argmax(pred, axis=1)
            d = {0:'Front', 1:'Rear', 2:'Side'}
            for key in d.keys():
                if pred_labels[0] == key:
                    st.write("Validating location of damage....Result:",d[key])
            st.write("Severity assessment complete.")
        else:
            st.write("Are you sure that your car is damaged? Please submit another picture of the damage.")
            st.write("Hint: Try zooming in/out, using a different angle or different lighting")
        

 
  else:
    st.warning('Please upload an image.')
        


main()
