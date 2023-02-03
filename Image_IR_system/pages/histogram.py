import streamlit as st
import numpy as np
from feature_extractors.histogram_ import Histogram
import cv2
import os
from PIL import Image

num_imgs=10
file_Empty=False

main_path=os.getcwd()
path=main_path.replace("\\","/")


path=[path+'/static/stanford_car_dataset/',path+"/static/csv_files/cars.csv"]
isFile = os.path.isfile(path[1])

if not isFile:
    with open(path[1], 'w') as creating_new_csv_file: 
        pass 
    print("Empty File Created Successfully")

file = open(path[1], "r")
file_content = file.read()
file.close()
 
st.header('Using HSV, histogram')
compare=st.sidebar.selectbox("Select the label you want to compare with",['Query label','Similar query labels'])
distance=st.selectbox("Select the distance",['chi-squared distance','Euclidean distance'])
num_imgs=st.sidebar.slider("No. of Similar Images")

if num_imgs ==0:
    num_imgs=10

query_image=st.file_uploader('Choose an image',type='.jpg')
submit=st.button('check for similar images')

if submit:
    if query_image is not None:
        file_bytes=np.asarray(bytearray(query_image.read()),dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)
        st.markdown('Query Image')
        st.image(opencv_image,channels="BGR")
        hist_obj=Histogram(path)

        if file_content == "":
            print('File found empty wait ... \n calculating features...')
            hist_obj.calculate_features()

        df_list=hist_obj.find_similar_images(opencv_image,distance)

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
            for i, (score,image_file) in enumerate(images):
                image=Image.open(image_file)
                cols[i].image(image)
        hist_obj.evaluate(num_imgs,df_list,query_image,compare)