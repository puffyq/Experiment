# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 20:57:38 2019

@author: puffy
"""
import os
import numpy as np
import random
import scipy.special as scipy
from scipy.spatial.distance import pdist
import pandas as pd
from sklearn import model_selection
from mvtpy.mvtest import mvtest
np.random.seed(1024)

dir_path = r'C:\data\linkPre_v2\linkPre\data_sto\subgraph'
save_dir = r'C:\data\linkPre_v2\linkPre\data_sto\train_in_mv'

def find_neg_mvIndex(link, number, iterate):
    test_mv = mvtest()
    index=[]
    for u in range(number):
        for f in range(u+1,number):
            temp=test_mv.test(link[u],link[f])
            if temp['p-value'][0] > 0.2:
                index.append([u,f])
            else:
                continue        
     
    neg_friend = np.array(random.sample(index,number))    
    neg_zeros = np.zeros(number)
    neg_case = np.c_[neg_friend[:,0]+1,neg_friend[:,1]+1, neg_zeros]
                
    return neg_case

def find_negtiveCase(a1_data, number, iterate):
    neg_all = {}
    for  i  in range(2,6):
        name  = str(iterate)+'_sub_a' + str(i) + '_'+str(number)+'.npy'
        file_path = os.path.join(dir_path,str(number),name)
        data = np.load(file_path)               
#        threshold = int(np.percentile(data[:,2],30))
        nolink_user = np.array(range(1,number+1))[~np.isin(range(1,number+1), list(set(data[:,0])))]         
#        nolink_user = np.array(range(1,number+1))[~np.isin(range(1,number+1), list(set(data[:,0][data[:,2]>threshold])))] 
        neg_all['a'+str(i)] = nolink_user
    
#    neg_user = set(neg_all['a2']) & set(neg_all['a3']) & set(neg_all['a4']) & set(neg_all['a5'])
    neg_user = set(neg_all['a2']) | set(neg_all['a3']) | set(neg_all['a4']) | set(neg_all['a5'])
    
    neg_user = list(neg_user)
    neg_friend = random.sample(range(1,number+1), 2*len(neg_user))
    neg_zeros = np.zeros(2*len(neg_user))
    neg_case = np.c_[2*neg_user,neg_friend, neg_zeros]

    return neg_case

def caculate_pu(data, number):
    
    prob_matrix = np.zeros((number,2))
    prob_matrix[:,0] = range(1,number+1)
    for u in range(number):
        u_num = sum(np.isin(data[:,0],u+1))
        u_prob = u_num/number
        prob_matrix[u,1] = u_prob
        
    return prob_matrix

def user_sample(index_0, prob_matrix, size):
    index_prob = np.zeros((len(index_0),1))
    sample = np.zeros((size,2))
    for u in range(number):
        index_prob[np.isin(index_0[:,0], u)] = prob_matrix[u,1]
    temp= pd.DataFrame(index_0)
    index = pd.DataFrame(list(range(len(index_0))))
    index_sample= index.sample(size,weights = index_prob[:,0])
    sample = temp.iloc[np.array(index_sample)[:,0]]
    sample = np.c_[np.array(sample)+1,np.zeros(len(sample))]
    return sample


def do_process(number,iterate):

#open the iterate_sub_a1_number.npy
    name = str(iterate)+'_sub_a1_'+str(number)+'.npy'
    file_path = os.path.join(dir_path,str(number),name)
#    file_path = "sub_a1_500.npy"
    data = np.load(file_path)
#    prob_matrix = caculate_pu(data, number)
    link = np.zeros((number,number))
    index = (np.array(data[:,0])-1,np.array(data[:,1])-1)
    link[index] = 1
#    index_0 = np.argwhere(link == 0)
    data_mv = find_neg_mvIndex(link, number, iterate)
#    data_0 = find_negtiveCase(data, number, iterate)
#    data_0 = user_sample(index_0, prob_matrix, size=int(0.5*number))
    data_all = np.r_[np.array(data),data_mv]
    data_all[:,2] = scipy.expit(data_all[:,2]) 
#    data_all[:,2][data_all[:,2]==0] = 0.2
    data_all[:,0:2] = data_all[:,0:2].astype(int) 
#save the train_in.npy    
    name_save = 'train_in_'+str(number)+'_'+str(iterate)
    file_save = os.path.join(save_dir,name_save)
    
    np.save(file_save,data_all)
    
if __name__ == "__main__":
    number = 500
#    iterate = 1
#    do_process(number, iterate)
    for i in range(1): 
        iterate = i+1 
        do_process(number,iterate)
    
    
#data_all = np.load("train_in.npy")
##data_all[:,2][data_all[:,2]>0.6] = 1
##data_all[:,2][data_all[:,2]<=0.6] = 0


