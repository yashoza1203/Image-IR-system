import streamlit as st
import numpy as np
from feature_extractors.vgg_ import VGG
import cv2
import os
from PIL import Image

st.header("using VGG")
compare=st.sidebar.selectbox("Select the label you want to compare with",['Query label','Similar query labels'])

num_imgs=st.sidebar.slider("No. of Similar Images")
if num_imgs ==0:
    num_imgs=10
query_image=st.file_uploader('Choose an image',type='.jpg')

main_path=os.getcwd()
path=main_path.replace("\\","/")

submit=st.button('check for similar images')

if submit:
    if query_image is not None:
        file_bytes=np.asarray(bytearray(query_image.read()),dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)
        st.markdown('Query Image')
        st.image(opencv_image,channels="BGR")
        vgg=VGG()
        VGG_list=vgg.get_sorted_list(query_image)
        vgg_list_top=VGG_list[:num_imgs]
        imgs=[]
        caption=[]

        st.markdown('Similar Images')
        view_images=[]
        n=5
        for i in range(0,len(vgg_list_top),n):
            view_images.append(vgg_list_top[i:i+n])
        
        for images in view_images:
            cols=st.columns(n)
            for i, (image_file,score) in enumerate(images):
                img=path+'/static/stanford_car_dataset/' + image_file + '.jpg'
                image=Image.open(img)
                cols[i].image(image,caption=image_file) 

        st.sidebar.text('Results')
        vgg.evaluate_model(num_imgs,VGG_list,query_image,compare)