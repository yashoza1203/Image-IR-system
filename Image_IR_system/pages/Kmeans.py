import streamlit as st
import numpy as np
import cv2
import os
from feature_extractors.kmeans_ import kMeans
from PIL import Image

st.markdown('Select your query image')
compare=st.sidebar.selectbox("Select the label you want to compare with",['Query label','Similar query labels'])
model=st.selectbox("Select the neural network model to extract features",['VGG','ResNet'])
query_image=st.file_uploader('Choose the image',type='.jpg')
submit=st.button('check for images')
main_path=os.getcwd()
path=main_path.replace("\\","/")


if submit:
    if query_image is not None:
        file_bytes=np.asarray(bytearray(query_image.read()),dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)
        st.markdown('Query Image')
        st.image(opencv_image,channels="BGR")
        km=kMeans()
        ftrs,filenames,feat=km.get_filename_features(model)
        km.fit_model(feat)
        grps=km.map_cluster_to_file(filenames)
        q_label=km.predict_image(query_image,ftrs)
        imgs=[]
        caption=[]
        st.markdown('Similar Images')
        view_images=[]
        n=5
        for i in range(0,len(grps[q_label[0]]),n):
            view_images.append(grps[q_label[0]][i:i+n])
        
        for images in view_images:
            cols=st.columns(n)
            for i, image_file in enumerate(images):
                img= path+'/static/stanford_car_dataset/' + image_file + '.jpg'
                image=Image.open(img)
                cols[i].image(image,caption=image_file) 
 
        st.sidebar.text('Results')
        km.evaluate(grps[q_label[0]],query_image,compare)