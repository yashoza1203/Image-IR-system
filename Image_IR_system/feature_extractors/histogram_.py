import streamlit as st
import cv2
import numpy as np
import os
import pickle as pkl
import scipy.spatial.distance
import csv
import glob

class Histogram():
    def __init__(self,path):
        self.ds=path
        main_path=os.getcwd()
        path=main_path.replace("\\","/")
        self.csv_file=self.ds[1]
        self.path=self.ds[0]
        self.dir_path=path + '/static/dir.pkl'

    def histogram(self,img,mask):
        hist=cv2.calcHist([img],[0, 1, 2], mask,(8,12,3),[0, 180, 0, 256, 0, 256])
        hist=cv2.normalize(hist,hist).flatten()
        return hist

    def describe(self,img):
        ##Converting the BGR image to HSV
        hsv_img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        features=[]
        #Grabbing the dimensions of image and calculating the center
        (h, w) = hsv_img.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        
        # (startX, endX, startY, endY) 
        segments=[(0,cX,0,cY),(cX,w,0,cY),(cX,w,cY,h),(0,cX,cY,h)]
        #         top-left     top right  bottom right bottom-left
        
        mask_img=np.zeros(img.shape[:2],dtype='uint8')
        (el_h,el_w)=(int(w*0.75)//2,int(h*0.75)//2)
        cv2.ellipse(mask_img,(cX,cY),(el_h,el_w),0,0,360,255,-1)
        
        for (startX, endX, startY, endY) in segments:
            ## construct a mask for each corner of the image, subtracting the elliptical center from it
            corner_mask=np.zeros(img.shape[:2],dtype='uint8')
            cv2.rectangle(corner_mask,(startX,startY),(endX,endY),255,-1)
            corner_mask=cv2.subtract(corner_mask,mask_img)
            hist=self.histogram(hsv_img,corner_mask)
            features.extend(hist)
        
        hist=self.histogram(hsv_img,mask_img)
        features.extend(hist)
        return features

    def chi2_distance(self,histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        return d

    def calculate_features(self):
        file=open(self.csv_file,'w')
        path = glob.glob(f"{self.path}/*.jpg")
        for img in path:
            imag = cv2.imread(img)
            features=self.describe(imag)
            features = [str(f) for f in features]
            file.write("%s,%s\n" % (img, ",".join(features)))
        file.close()

    def evaluate(self,limit,diff_list,path,compare):
        new_list=diff_list
        path=path.name
        yo_list=new_list[:limit]

        with open(self.dir_path,'rb') as f:
            dir_list=pkl.load(f)

        def get_labels(path):
            file_name = os.path.basename(path) 
            dir_name=dir_list[file_name] 
            dir_query=dir_name[0]
            dir_q_name=dir_query.split('/')
            return dir_q_name[1]

        def top_labels(yo_list):
            label_results=[]
            for score,path in yo_list:
            # for path in yo_list:
                label=get_labels(path)
                label_results.append(label)
            return label_results

        def get_count_qry(q_car_name,top_results):
            count=0
            for i in range(len(top_results)):
                if top_results[i]==q_car_name:
                    count+=1
            return count

        def get_count_similar_qry(q_car_name,top_results):   
            count=0
            for i in top_results:
                if i in q_car_name:
                    count+=1
            return count

        def get_similar_labels(query_label):
            first_word_qlabel=query_label.split()[0]
            lbl_st=[] 
            for i in set_label:
                if first_word_qlabel==i.split()[0]:
                    lbl_st.append(i)
            return lbl_st

        total_label=top_labels(diff_list)
        set_label=set(total_label)

        query_label=get_labels(path)
        similar_query_labels= get_similar_labels(query_label)
        top_label=top_labels(yo_list)

        guesses=[0]*limit
        if compare== 'Query label':
            count =get_count_qry(query_label,top_label)
        else:
            count =get_count_similar_qry(similar_query_labels,top_label)

        total_label=top_labels(diff_list)
        total_count=0
        for words in similar_query_labels:
            total_count=total_count+total_label.count(words)
        
        def calculate_f1_score(precision,recall):
            f1_score=2*(precision*recall)/(precision+recall)
            return f1_score

        def calculate_precision(guesses):
            relevant_items_retrieved=guesses.count(1)
            retrieved_items=len(guesses)    
            precision =relevant_items_retrieved/retrieved_items
            return precision

        def calculate_recall(guesses):
            relevant_items_retrieved=guesses.count(1)
            q_count=total_label.count(query_label)
            if compare=='Query label':
                relevant_items=q_count
            else:
                relevant_items=total_count
            recall = relevant_items_retrieved/relevant_items
            return recall

        for i in range(count): guesses[i]=1
        precision=calculate_precision(guesses)
        recall=calculate_recall(guesses)
        st.sidebar.text(f"precision : {precision}%")
        st.sidebar.text(f"recall : {recall}%")
        st.sidebar.text(f"f1 score  : {calculate_f1_score(precision,recall)}%")
        
    def search(self,query_features,distance):
        results={}
        with open(self.csv_file) as f:
            reader=csv.reader(f)
            image_similarity_measure = scipy.spatial.distance.euclidean
            for row in reader:
                features=[float(x) for x in row[1:]]
                if distance=='chi-squared distance':
                    d=self.chi2_distance(query_features,features)
                else:
                    d=image_similarity_measure(query_features,features)
                results[row[0]] = d
            f.close()
        results=sorted([(v,k) for (k,v) in results.items()])
        return results

    def find_similar_images(self,query_img,distance):
        queryfeatures=self.describe(query_img)
        results=self.search(queryfeatures,distance)
        return results