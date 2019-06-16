# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 20:57:38 2019

@author: puffy


"""
import os
import xuan_a8
#import log_svd
#import no_cir
import xuan_svd
import numpy as np
import random
import scipy.special as scipy
import pandas as pd
from sklearn import model_selection
np.random.seed(1024)


dir_path = r"C:\data\linkPre_v2\linkPre\data_sto\train_in"
effective_path = r"C:\data\linkPre_v2\linkPre\data_sto\effective"


def completion(matrix,maxsize=10):
    if len(matrix) < maxsize:
        c_matrix = np.zeros(((maxsize-len(matrix)),matrix.shape[1]))
        matrix = np.r_[matrix,c_matrix]    
    return matrix

def open_data(number,iterate):
    
    name = "train_in_"+str(number)+"_"+str(iterate)+'.npy'
    file_path = os.path.join(dir_path,name)
    data_all = np.load(file_path)   
    effective_dir = os.path.join(effective_path,str(number))
    file_box = os.listdir(effective_dir)
    file_effective = []
    for i in file_box:
        if str(iterate)+'_a' in i:
            file_effective.append(os.path.join(effective_dir,i))

    adj_matrix = dict()

    for i in file_effective: 
        name = i.split('\\')[-1].split('.')[0]
        if ('s' in name)|('t' in name):
            maxsize = 10
        else:
            maxsize =number            
        adj_matrix[name] = completion(np.load(i),maxsize)

    train, test = model_selection.train_test_split(data_all,random_state=int(0.1*len(data_all))) #0.3*N
    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    return train,test,adj_matrix
    

def do_process(length, train,test,adj_matrix,number,iterate):
    
#    model_it = log_svd.log_svd()
    model_it = xuan_a8.xuan()
    matrix = model_it.iterate(length, number, iterate, adj_matrix, train, test, 10)
#length, 
    return matrix

def do_process_svd(length, train,test,number,iterate):
#    
#    name = "train_in_"+str(number)+"_"+str(iterate)+'.npy'
#    file_path = os.path.join(dir_path,name)
#    data_all = np.load(file_path)   
#
#    train, test = model_selection.train_test_split(data_all,random_state=int(0.3*len(data_all))) #0.3*N
#    train = pd.DataFrame(train)
#    test = pd.DataFrame(test)
 
    model_it = xuan_svd.xuan_svd()
    matrix = model_it.iterate(length, number, train,test,10)
    
    return matrix

if __name__ == '__main__':
    
    number = 500
    length = 50
    inde = 1
    
#    iterate = 10
    auc_all_fuse = 0
    precision_fuse = 0
    recall_fuse = 0
    f1_fuse = 0
    auc_all_gssvd = 0
    precision_gssvd = 0
    recall_gssvd = 0
    precision_base = 0  
    recall_base = 0
    f1_base = 0
    f1_gssvd = 0
    precision_train = 0
    recall_train = 0
    f1_train = 0
    length_all = 0
    res = {}
    
    for iterate in range(1,inde+1):        
  
        train,test,adj_matrix = open_data(number,iterate)
        matrix_fuse = do_process(length, train,test,adj_matrix,number,iterate)   
        print (str(iterate)+' fuse done!')
        evaluate_matrix_fuse = matrix_fuse[-2]
        base_matrix_fuse = matrix_fuse[-1]
        evaluate_marix_train = matrix_fuse[-3]
        length_all =length_all + matrix_fuse [-4]
        auc_all_fuse = auc_all_fuse + evaluate_matrix_fuse['auc_value']
        precision_fuse = precision_fuse + evaluate_matrix_fuse['precision']
        recall_fuse = recall_fuse + evaluate_matrix_fuse['recall']
        f1_fuse = f1_fuse + evaluate_matrix_fuse['f1']
        precision_base = precision_base + base_matrix_fuse['precision_base']
        recall_base = recall_base + base_matrix_fuse['recall_base']
        f1_base = f1_base + base_matrix_fuse['f1_base']
        precision_train = precision_train + evaluate_marix_train['precision']
        recall_train = recall_train + evaluate_marix_train['recall']
        f1_train = f1_train + evaluate_marix_train['f1']
        
        matrix_gssvd = do_process_svd(length, train,test,number,iterate) 
        print (str(iterate)+' gssvd done!')
        evaluate_matrix_gssvd = matrix_gssvd[-2]
#        base_matrix_gssvd = matrix_gssvd[-2]
        auc_all_gssvd = auc_all_gssvd + evaluate_matrix_gssvd['auc_value']
        precision_gssvd = precision_gssvd+ evaluate_matrix_gssvd['precision']
        recall_gssvd = recall_gssvd + evaluate_matrix_gssvd['recall']
        f1_gssvd = f1_gssvd + evaluate_matrix_gssvd['f1_gssvd']
        
    res['auc_mean_fuse'] = float(auc_all_fuse)/inde
    res['precision_mean_fuse'] = float(precision_fuse)/inde
    res['recall_mean_fuse'] = float(recall_fuse)/inde
    res['f1_mean_fuse'] = float(f1_fuse)/inde
    
    res['auc_mean_gssvd'] = float(auc_all_gssvd)/inde
    res['precision_mean_gssvd'] = float(precision_gssvd)/inde
    res['recall_mean_gssvd'] = float(recall_gssvd)/inde
    res['f1_mean_gssvd'] = float(f1_gssvd)/inde
    
    res['precision_base_mean'] = float(precision_base)/inde
    res['recall_base_mean'] = float(recall_base)/inde
    res['f1_mean_base'] = float(f1_base)/inde

    res['precision_mean_train'] = float(precision_train)/inde
    res['recall_mean_train'] = float(recall_train)/inde
    res['f1_mean_train'] = float(f1_train)/inde
    
    res['length'] = float(length_all)/inde
    
    