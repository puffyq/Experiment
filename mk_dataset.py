# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:31:33 2019

@author: puffy
"""
import os
from snowballsampling import randomseed
from snowballsampling import snowballsampling
#from snowballsampling import surroundings
import networkx as nx
import pandas as pd
import numpy as np

dir_path = r"C:\data\linkPre_v2\linkPre\data_sto\subgraph"
dir_path_subgraph = r"C:\data\linkPre_v2\linkPre\data_sto\index"

names = ['userID','friendID','link']
file_a1 = r"C:\data\linkPre_v2\linkPre\YouTube-dataset\data\1-edges.csv"
file_a2 = r"C:\data\linkPre_v2\linkPre\YouTube-dataset\data\2-edges.csv"
file_a3 = r"C:\data\linkPre_v2\linkPre\YouTube-dataset\data\3-edges.csv"
file_a4 = r"C:\data\linkPre_v2\linkPre\YouTube-dataset\data\4-edges.csv"
file_a5 = r"C:\data\linkPre_v2\linkPre\YouTube-dataset\data\5-edges.csv"
data_a1 = pd.read_csv(file_a1,sep=',',names = names)
data_a2 = pd.read_csv(file_a2,sep=',',names = names)
data_a3 = pd.read_csv(file_a3,sep=',',names = names)
data_a4 = pd.read_csv(file_a4,sep=',',names = names)
data_a5 = pd.read_csv(file_a5,sep=',',names = names)

G=nx.MultiGraph()

G.add_edges_from(data_a1.loc[:,['userID','friendID']].values)
G.add_edges_from(data_a2.values)
G.add_edges_from(data_a3.values)
G.add_edges_from(data_a4.values)
G.add_edges_from(data_a5.values)

def save(dir_path,number,flag,sub_a):
    subDir_path = os.path.join(dir_path,str(number))
    if os.path.exists(subDir_path) == False:
        os.mkdir(subDir_path)
        
    for i in range(5):
        name = str(flag)+"_"+"sub_a"+str(i+1)+"_"+str(number)
        name_var = "sub_a"+str(i+1)+"_new"
        file_path = os.path.join(subDir_path,name)
        np.save(file_path,sub_a[name_var])

def modify_index(list_c,uniset):
    user = list(set(list_c))
    list_new = np.array(list_c.copy())
    for i in user:
        index = np.where(uniset == i)
        list_new[list_c == i] = index[0]+1

    return list_new

def form_newMatrix(subUser,subFriend,subLink,uniset):
    new_subUser = np.array(modify_index(subUser,uniset))
    new_subFriend = np.array(modify_index(subFriend,uniset))
    new_matrix = np.c_[new_subUser,new_subFriend,np.array(subLink)]
    return new_matrix


def find_newFriend(sub_user,sub_friend,sub_link):
    userset = list(set(sub_user))
    sub_matrix = []
    index = np.isin(sub_friend,userset)
    friend = np.array(sub_friend)[index]
    user = np.array(sub_user)[index]
    link = np.array(sub_link)[index]
    sub_matrix = np.c_[user,friend,link]
    return sub_matrix

def save_subgraph(dir_path,number,flag,subgraph):
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)
        
    name = 'subgraph_'+str(number)+'_'+str(flag)
    file_path = os.path.join(dir_path,name)
    np.save(file_path,subgraph)
    

def find_sub_net(network, subgraph):
    subnet = network[network['userID'].isin(subgraph)]
    return [list(subnet[x]) for x in subnet.columns]

def do_process(number,iterate):        
    seed = [randomseed(G)]
    #seed = [2661]
    subgraph = list(snowballsampling(G,seed,number))
    #neighbors = surroundings(G,subgraph)
#    subgraph = np.load(r"C:\data\linkPre_v2\linkPre\data_sto\index\subgraph_2000_10.npy")
    save_subgraph(dir_path_subgraph,number,iterate,subgraph)    
    
    sub_user_a1,sub_friend_a1,sub_link_a1 = find_sub_net(data_a1, subgraph)
    sub_user_a2,sub_friend_a2,sub_link_a2 = find_sub_net(data_a2, subgraph)
    sub_user_a3,sub_friend_a3,sub_link_a3 = find_sub_net(data_a3, subgraph)
    sub_user_a4,sub_friend_a4,sub_link_a4 = find_sub_net(data_a4, subgraph)
    sub_user_a5,sub_friend_a5,sub_link_a5 = find_sub_net(data_a5, subgraph)
    
    sub_a1_newIndex = find_newFriend(sub_user_a1,sub_friend_a1,sub_link_a1)
    sub_a2_newIndex = find_newFriend(sub_user_a2,sub_friend_a2,sub_link_a2)
    sub_a3_newIndex = find_newFriend(sub_user_a3,sub_friend_a3,sub_link_a3)
    sub_a4_newIndex = find_newFriend(sub_user_a4,sub_friend_a4,sub_link_a4)
    sub_a5_newIndex = find_newFriend(sub_user_a5,sub_friend_a5,sub_link_a5)
    
    uniset = set(sub_user_a1) | set(sub_user_a2) | set(sub_user_a3) | set(sub_user_a4) | set(sub_user_a5)
    uniset = np.array(list(uniset))
    
    sub_a1_new = form_newMatrix(sub_a1_newIndex[:,0],sub_a1_newIndex[:,1],sub_a1_newIndex[:,2],uniset)
    sub_a2_new = form_newMatrix(sub_a2_newIndex[:,0],sub_a2_newIndex[:,1],sub_a2_newIndex[:,2],uniset)
    sub_a3_new = form_newMatrix(sub_a3_newIndex[:,0],sub_a3_newIndex[:,1],sub_a3_newIndex[:,2],uniset)
    sub_a4_new = form_newMatrix(sub_a4_newIndex[:,0],sub_a4_newIndex[:,1],sub_a4_newIndex[:,2],uniset)
    sub_a5_new = form_newMatrix(sub_a5_newIndex[:,0],sub_a5_newIndex[:,1],sub_a5_newIndex[:,2],uniset)

    sub_a = dict()
    sub_a['sub_a1_new'] = sub_a1_new
    sub_a['sub_a2_new'] = sub_a2_new
    sub_a['sub_a3_new'] = sub_a3_new
    sub_a['sub_a4_new'] = sub_a4_new
    sub_a['sub_a5_new'] = sub_a5_new

    print("strat saving.....")    
    save(dir_path,number,iterate,sub_a)        

if __name__=='__main__':
    number = 4000   
    for i in range(0,10):
        iterate = i+1
        do_process(number,iterate)
#    np.save("sub_a1_500",sub_a1_new)
#    np.save("sub_a2_500",sub_a2_new)
#    np.save("sub_a3_500",sub_a3_new)
#    np.save("sub_a4_500",sub_a4_new)
#    np.save("sub_a5_500",sub_a5_new)

