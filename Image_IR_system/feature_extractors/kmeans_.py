from sklearn.cluster import KMeans
import os
import numpy as np
import pickle as pkl
import streamlit as st

class kMeans():        
    def __init__(self) :
        self.kmeans=KMeans(init = "k-means++",n_clusters=47,n_init = 12,random_state=45)
        main_path=os.getcwd()
        self.path=main_path.replace("\\","/")
        self.dir_path=self.path + '/static/dir.pkl'

    def get_filename_features(self,model):
        if model=='VGG':
            with open(self.path+'/static/vgg_features.pkl','rb') as f:  
                features=pkl.load(f)
        else:
            with open(self.path+'/static/resnet50_features.pkl','rb') as f:  
                features=pkl.load(f)

        filenames=np.array(list(features.keys()))
        feats=np.array(list(features.values()))
        feat=feats.reshape(-1,feats.shape[-1])
        return features,filenames,feat

    def fit_model(self,feat):
        self.kmeans.fit(feat)

    def map_cluster_to_file(self,filenames):
        groups={}
        for File,cluster in zip(filenames,self.kmeans.labels_):
            if cluster not in groups.keys():
                groups[cluster]=[]
                groups[cluster].append(File)
            else:
                groups[cluster].append(File)
        return groups
    
    def predict_image(self,query,features):
        qlabel=query.name
        query_label=qlabel.split('.')[0]
        q_label=self.kmeans.predict(features[query_label])
        return q_label

    def evaluate(self,diff_list,path,compare):
        path=path.name
        new_list=diff_list
        for i in range(len(diff_list)):
            new_list[i]=new_list[i]+'.jpg'
        yo_list=new_list

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
            for path in yo_list:
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

        guesses=[0]*len(diff_list)

        if compare== 'Query label':
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
            if compare=='Query label':
                relevant_items=41
            else:
                relevant_items=151
            recall = relevant_items_retrieved/relevant_items
            return recall

        for i in range(count): guesses[i]=1
        precision=calculate_precision(guesses)
        recall=calculate_recall(guesses)
        st.sidebar.text(f"precision : {precision}%")
        st.sidebar.text(f"recall : {recall}%")
        st.sidebar.text(f"f1 score  : {calculate_f1_score(precision,recall)}%")
