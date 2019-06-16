# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:33:05 2019

@author: puffy
"""
import numpy as np
import random  

def caculate_cn(number, data):
    
    data = np.array(data)
    data = data[~(np.isin(data[:,2],0.5))]
    user_list = list(range(1,number+1))
    rec_res = {}
    for u in user_list:        
        rec_res[u] = set(data[:,1][np.isin(data[:,0],u)].astype(int)) #| set(data[:,0][np.isin(data[:,1],u)].astype(int))

    cn_all = {}
    for u,u_values in rec_res.items():
        cn_list = []
        for i,i_values in rec_res.items():
            cn = len(u_values & i_values)/np.sqrt(len(u_values)*len(i_values))
            cn_list.append(cn)
        cn_all[u] = cn_list
    
    return cn_all

def auc(test, y_hat, n = 200):
    linked = np.where(test[2] >= 0.6)
    noLink = np.where(test[2] <= 0.5)    
#    linked = np.where(test[2] == 1)
#    noLink = np.where(test[2] == 0) 
    n_dot = 0
    n_ddot = 0 
    i=0
    while i < n:
        i = i+1
        index_link = random.randint(0,len(linked[0])-1)  
        index_nolink = random.randint(0,len(noLink[0])-1)  
        if y_hat[linked[0][index_link]] > y_hat[noLink[0][index_nolink]]:
            n_dot = n_dot + 1
            continue
        elif y_hat[linked[0][index_link]] == y_hat[noLink[0][index_nolink]]:
            n_ddot =n_ddot + 1
            continue
        else:
            continue
        
    
    auc = (n_dot+0.5*n_ddot)/n    
    return auc

def precision_recall(recom_base, test_box, recom_res, user_box):
    commen_base = 0
    commen_num = 0
    r_total = 0
    t_total = 0
    t_btotal = 0
    for u in user_box:
        r = list(test_box[u])
        t_base = recom_base[u]
        t = recom_res[u]
        com_base = len(set(r) & set(t_base[:,0]))
        com_temp = len(set(r) & set(t))
        commen_base = commen_base + com_base
        commen_num = commen_num + com_temp
        r_total = r_total + len(r)
        t_total = t_total + len(t)
        t_btotal = t_btotal + len(t_base)
    
    precision_base = float(commen_base)/float(r_total)
    precision = float(commen_num)/float(r_total)
    recall_base = float(commen_base)/float(t_btotal)
    recall = float(commen_num)/float(t_total)
    
    return precision,recall,precision_base,recall_base

def f_value(precision, recall):
    if (precision==0.0)|(recall==0.0):
        f1 = 0
    else:
        f1 = (2*precision*recall)/(precision+recall)
    return f1

def pre_cn(test, cn_all, k):
    
    test = np.array(test)
    test = test[~(np.isin(test[:,2],0.5))]
    rec_res = {}
    test_user = set(test[:,0])
    for u in test_user:
        temp = np.array(cn_all[int(u)])
        sort_cn = np.sort(temp[~np.isnan(temp)])
        if sum(sort_cn > 0) <= k:
            if len(sort_cn[sort_cn>0]) == 0:
                thre = 2
            else:
                thre = min(sort_cn[sort_cn>0])
        else:
            thre = sort_cn[:-1][-k]
        recom_item  = np.where((temp > thre) & (temp != 1))
        recom_list = np.zeros((k,1),dtype = np.int)
        recom_list[:len(recom_item[0]),0] = recom_item[0]+1
        rec_res[u] = recom_list
        
    return rec_res


def top_k(data, test, cn_all, user_group, item_group, U1_init, U2_init, U3_init, U4_init, k):
    number = max(data[1])#500
    test = test[~(np.isin(test[2],0.5))]
    data = data[~(np.isin(data[2],0.5))]
    user_box = list(set(test[0]))
    recom_res = {}
    test_box = {}
    for u in user_box:
        test_box[u] = data[1][np.isin(data[0],u)].tolist()
        user = [u-1]*number
        item = list(range(number))
        y_hat = np.sum(np.multiply(U1_init[user] +  \
                                   U3_init[np.int64(user_group[user]) - 1],
                                   U2_init[item] +  \
                                   U4_init[np.int64(item_group[item]) - 1]),
                     axis=1)
        
        sort_y = np.sort(y_hat)       
        thre = sort_y[-k]
        recom_item  = np.where(y_hat >= thre)
        recom_res[u] = recom_item[0]+1
    recom_base = pre_cn(test, cn_all, k) 
    
    precision,recall,precision_base,recall_base = precision_recall(recom_base, test_box, recom_res, user_box)
    f1 = f_value(precision, recall)
    f1_base = f_value(precision_base, recall_base)
    
    return precision,recall,precision_base,recall_base,f1,f1_base
