import streamlit as st
import numpy as np
from feature_extractors.sift_ import SIFT
import cv2
import os
from PIL import Image

main_path=os.getcwd()
path=main_path.replace("\\","/")
dataset=path+'/static/stanford_car_dataset/'

st.header('Using SIFT (Scale Invariant Fourier Transform)')
compare=st.sidebar.selectbox("Select the label you want to compare with",['Query label','Similar query labels'])
matcher=st.selectbox("Select the matcher",['bf matcher','flann matcher'])
query_image=st.file_uploader('Choose an image',type='.jpg')
submit=st.button('check for similar images')
num_imgs=st.sidebar.slider("No. of Similar Images")

if num_imgs ==0:
    num_imgs=10


if submit:
    if query_image is not None:
        file_bytes=np.asarray(bytearray(query_image.read()),dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)
        st.markdown('Query Image')
        st.image(opencv_image,channels="BGR")
        sift_obj=SIFT(dataset)
        if matcher=='sift matcher':
            df_list=sift_obj.get_scores_bf_matcher(query_image)
        else:
            df_list=sift_obj.get_scores_flann_matcher(query_image)

        imgs=[]
        caption=[]
        final_list=df_list[:num_imgs]
        st.markdown('Similar Images')
        view_images=[]
        n=5
        for i in range(0,len(final_list),n):
            view_images.append(final_list[i:i+n])
        
        for images in view_images:
            cols=st.columns(n)
            for i, (image_file,score) in enumerate(images):
                image=Image.open(image_file)
                cols[i].image(image,caption=score) 

        sift_obj.evaluate(df_list,num_imgs,query_image,compare)