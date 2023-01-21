# Importing the libraries
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from utils import config
import numpy as np
import argparse
import cv2
import os
from PIL import Image
modelName = 'weights-001-0.4708.hdf5'
modelPath = config.outputPath + '/' + modelName
model = load_model("./weights-001-0.4708.hdf5")
import streamlit as st

from tensorflow.keras.preprocessing import image
def load_image(path):
    # global COUNT
    img = image.load_img(path,target_size=model.input_shape[1:3]) 
    x = image.img_to_array(img) 
    x = np.expand_dims(x,axis=0) 
    x = preprocess_input(x) 
    # COUNT=COUNT-1
    return img,x
COUNT=0
def header(url):
     st.markdown(f'<div style="text-align:center"><p style="background-color:#FF0000;color:#FFFFFF;font-size:24px;border-radius:2%;">{url}</p></div', unsafe_allow_html=True)
st.title('Benin or Malignent Breast Cancer Classifier')
image_file = st.file_uploader("Upload Image",type=["csv","png","jpg","jpeg"])
if image_file is not None:
    
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    st.write(file_details)
    img = load_image(image_file)
    #st.image(img,height=250,width=250)
    temp=str(COUNT)
    try:
        with open(os.path.join("tempDir",temp),"wb") as f: 
            f.write(image_file.getbuffer()) 
    except: 
        st.write('An Error Has Occured please try again with another file.')           
    st.success("Saved File")
    if st.button('Predict'):
        
        img = Image.open('tempDir/{}'.format(COUNT))

        try:
            # img.save('static/{}.png'.format(COUNT))
            img_arr = cv2.imread('tempDir/{}'.format(COUNT))

            #img_arr = cv2.resize(img_arr, (128,128))
            #img_arr = img_arr / 255.0
            #img_arr = img_arr.reshape(1, 128,128,3)
           # prediction = model.predict(img_arr)

            # Convert it from BGR to RGB channel ordering, then Resize it to 48x48, 
            # and preprocess it
            image1 = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            image1 = cv2.resize(image1, (48, 48))
            image1 = img_to_array(image1)

            ## The image is now represented by a NumPy array of shape (48, 48, 3), however 
            # we need the dimensions to be (1, 3, 48, 48) so we can pass it
            # through the network and then we'll also preprocess the image by subtracting the 
            # mean RGB pixel intensity from the ImageNet dataset

            image1 = preprocess_input(image1)
            #image1 /=  255.0

            image1 = np.expand_dims(image1, axis=0)

            # Pass the image through the model to determine if the person has malignant
            (benign, malignant) = model.predict(image1)[0]


            # Determine the class label and so the color we will use to add text
            label = "benign" if benign > malignant else "malignant"
            color = (0, 255, 0) if label == "benign" else (0, 0, 255)

            # Adding the probability in the label
            label = "{}: {:.2f}%".format(label, max(benign, malignant) * 100)
        
            # Displaying the label on the output image
            # cv2.putText(image, label, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
                    
            # Showing the output image
            #st.danger('Cancer Type & Percentage : '+label)
            header(label)
           # cv2.imshow(label,img)
            # cv2.waitKey(0)
            
        except:
            st.write('An Error Has Occured please try again with another file.') 