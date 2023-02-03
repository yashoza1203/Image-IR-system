import cv2
import streamlit as st
import numpy as np
import os
import pickle as pkl
import glob

class SIFT():
    def __init__(self,dataset):
        # Create SIFT Object
        self.sift = cv2.SIFT_create()
        main_path=os.getcwd()
        path=main_path.replace("\\","/")
        self.ds_path=dataset
        self.images =glob.glob(f"{dataset}/*.jpg")
        self.dir_path=path + '/static/dir.pkl'

    def SIFT_based_matcher(self,query_img,sample_img):
        # find the keypoints and descriptors with SIFT
        _, des1 = self.sift.detectAndCompute(query_img,None)
        _, des2 = self.sift.detectAndCompute(sample_img,None)
        
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        
        # Apply ratio test
        good=[]
        for match1,match2 in matches:
            if match1.distance < 0.75*match2.distance:
                good.append([match1])
        return len(good)
    

    def flann_based_matcher(self,query_img,sample_img):
        # find the keypoints and descriptors with SIFT
        _, des1 = self.sift.detectAndCompute(query_img,None)
        _, des2 = self.sift.detectAndCompute(sample_img,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)  

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        good = []
        # ratio test
        for i,(match1,match2) in enumerate(matches):
            if match1.distance < 0.7*match2.distance:
                good.append([match1])
        return len(good)

    def get_scores_bf_matcher(self,qpath):
        bf_list=[]
        qname=qpath.name
        qpath=self.ds_path+ qname
        query=cv2.imread(qpath)
        for img in self.images:
            sample_img=cv2.imread(img)
            good_len=self.SIFT_based_matcher(query,sample_img)
            bf_list.append([img,good_len])
        bf_list.sort(key = lambda x: x[1],reverse=True)
        return bf_list

    def get_scores_flann_matcher(self,qpath):
        flan_list=[]
        qname=qpath.name
        qpath=self.ds_path+ qname
        query=cv2.imread(qpath)
        for img in self.images:
            sample_img=cv2.imread(img)
            good_len=self.flann_based_matcher(query,sample_img)
            flan_list.append([img,good_len])
        flan_list.sort(key = lambda x: x[1],reverse=True)
        return flan_list

    def evaluate(self,main_list,limit,qpath,compare):
        df_list=main_list[:limit]
        qpath=qpath.name
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
            for path,score in yo_list:
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

        total_label=top_labels(main_list)
        set_label=set(total_label)

        query_label=get_labels(qpath)
        similar_query_labels= get_similar_labels(query_label)
        top_label=top_labels(df_list)
     
        if compare== 'Query label':
            count_df =get_count_qry(query_label,top_label)
        else:
            count_df =get_count_similar_qry(similar_query_labels,top_label)

        total_count=0
        for words in similar_query_labels:
            total_count=total_count+total_label.count(words)
        q_count=total_label.count(query_label)
        guesses=[0]*limit

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
            if compare=='Query label':
                relevant_items=q_count
            else:
                relevant_items=total_count
            recall = relevant_items_retrieved/relevant_items
            return recall

        for i in range(count_df): guesses[i]=1
        precision=calculate_precision(guesses)
        recall=calculate_recall(guesses)
        st.sidebar.text(f"precision : {precision}%")
        st.sidebar.text(f"recall : {recall}%")
        st.sidebar.text(f"f1 score : {calculate_f1_score(precision,recall)}%")