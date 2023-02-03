import os
import pickle as pkl
import numpy as np
import streamlit as st

class VGG():
    def __init__(self) :
        main_path=os.getcwd()
        self.path=main_path.replace("\\","/")

        with open(self.path+'/static/vgg_features.pkl','rb') as f:  
            self.vgg_features=pkl.load(f)
        self.image_ids=list(self.vgg_features.keys())
        self.dir_path=self.path + '/static/dir.pkl'

    def evaluate_model(self,limit,diff_list,qpath,compare):
        path=qpath.name
        new_list=diff_list
        for i in range(len(new_list)):
            new_list[i][0]=new_list[i][0]+'.jpg'
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
            for path,score in yo_list:
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
        
        if compare=='Query label':
            count =get_count_qry(query_label,top_label)
        else:
            count =get_count_similar_qry(similar_query_labels,top_label)

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
        st.sidebar.text(f"f1 score : {calculate_f1_score(precision,recall)}%")

    def get_sorted_list(self,query):
        diff_list=[]
        query=query.name
        img_id=query.split('.')[0]
        qftrs=self.vgg_features[img_id]
        qftrs=qftrs / np.linalg.norm(qftrs)
        for id in self.image_ids:
            ft=self.vgg_features[id] / np.linalg.norm(self.vgg_features[id])
            dists = np.linalg.norm(ft-qftrs, axis=1)
            diff_list.append([id,dists])
        diff_list.sort(key = lambda x: x[1])
        return diff_list