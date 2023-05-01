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
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm



model = tf.keras.models.load_model(r"dmg_car-weights-CNN.h5")
explainer = lime_image.LimeImageExplainer()

# Define the segmenter function for better explanations
segmenter = SegmentationAlgorithm('slic', n_segments=50, compactness=10, sigma=1)
#
def predict_image(img, model, threshold):
    # Load image and preprocess it
    img = load_img(img, target_size=model.input_shape[1:4])
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    # Make predictions using the model
    prediction = model.predict(img_array)[0][0]
    
    # Check if the prediction is above the threshold
    if prediction >= threshold:
        result = 'damaged'
    else:
        result = 'not damaged'
    
    
    # Generate LIME explanation
    explanation = explainer.explain_instance(img_array[0], model.predict, segmentation_fn=segmenter)
    top_labels = prediction
    explanation_html = explanation.as_html(top_labels=top_labels)
    
    # Return prediction and explanation as a dictionary
    return {'prediction': prediction, 'explanation_html': explanation_html}





#
st.title('Car Damage Predictor')
st.markdown('Upload an image of a car to see if it is damaged or not.')
st.sidebar.title('Settings')
threshold = st.sidebar.slider('Threshold', 0.0, 1.0, 0.5, 0.01)
def main():
  uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
  #
  if st.button('Predict'):
        # Check if user has uploaded a file
        if uploaded_file is not None:
            # Make prediction and generate explanation using LIME
            result = predict_image(model, uploaded_file, threshold)
            prediction = result['prediction']
            explanation_html = result['explanation_html']

            # Display prediction and explanation
            if prediction[0] >= threshold:
                st.success('The car is undamaged with a probability of {}%'.format(round(prediction[0]*100, 2)))
            else:
                st.error('The car is damaged with a probability of {}%'.format(round(prediction[1]*100, 2)))
            st.markdown('## Explanation')
            st.write(explanation_html, unsafe_allow_html=True)
        else:
            st.warning('Please upload an image.')
  #


if __name__ == "__main__":
    main()
