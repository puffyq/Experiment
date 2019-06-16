# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 08:45:09 2019

@author: puffy
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

names = ['userID','friendID','link']
file_a1 = r"E:\linkpre\linkPre\YouTube-dataset\data\1-edges.csv"
file_a2 = r"E:\linkpre\linkPre\YouTube-dataset\data\2-edges.csv"
file_a3 = r"E:\linkpre\linkPre\YouTube-dataset\data\3-edges.csv"
file_a4 = r"E:\linkpre\linkPre\YouTube-dataset\data\4-edges.csv"
file_a5 = r"E:\linkpre\linkPre\YouTube-dataset\data\5-edges.csv"
data_a1 = pd.read_csv(file_a1,sep=',',names = names)
data_a2 = pd.read_csv(file_a2,sep=',',names = names)
data_a3 = pd.read_csv(file_a3,sep=',',names = names)
data_a4 = pd.read_csv(file_a4,sep=',',names = names)
data_a5 = pd.read_csv(file_a5,sep=',',names = names)

#G1=nx.Graph()
#G1.add_nodes_from(np.arange(1,max(data_a1['friendID'])))#加列表中的点
#G1.add_weighted_edges_from(np.array(data_a1))
G=nx.MultiGraph()

G.add_edges_from(data_a1.loc[:,['userID','friendID']].values)
G.add_edges_from(data_a2.values)
G.add_edges_from(data_a3.values)
G.add_edges_from(data_a4.values)
G.add_edges_from(data_a5.values)
#G.add_nodes_from(np.arange(1,number))#加列表中的点
    

ave_path = 0
for c in sorted(nx.connected_components(G1),key=len,reverse=True):
    ave_path = ave_path + len(c)

ave_path = ave_path/len(list(nx.connected_components(G1)))        
sup_infpath = len(max(nx.connected_components(G1),key=len))
sum_nodes = len(G1.nodes())
sum_edges = len(G1.edges())
ave_degree = np.mean(G1.degree())
ave_cluster = nx.average_clustering(G1)

#################plot####################

G1=nx.Graph()
G1.add_nodes_from(np.arange(1,max(data_a1['friendID'])))#加列表中的点
G1.add_weighted_edges_from(np.array(data_a1))

G2=nx.Graph()
G2.add_nodes_from(np.arange(1,max(data_a1['friendID'])))#加列表中的点
G2.add_weighted_edges_from(np.array(data_a2))

G3=nx.Graph()
G3.add_nodes_from(np.arange(1,max(data_a1['friendID'])))#加列表中的点
G3.add_weighted_edges_from(np.array(data_a3))

G4=nx.Graph()
G4.add_nodes_from(np.arange(1,max(data_a1['friendID'])))#加列表中的点
G4.add_weighted_edges_from(np.array(data_a4))

G5=nx.Graph()
G5.add_nodes_from(np.arange(1,max(data_a1['friendID'])))#加列表中的点
G5.add_weighted_edges_from(np.array(data_a5))

degree_1 = nx.degree_histogram(G1)
degree_2 = nx.degree_histogram(G2)
degree_3 = nx.degree_histogram(G3)
degree_4 = nx.degree_histogram(G4)
degree_5 = nx.degree_histogram(G5)


plt.subplot(2,1,1)
x_1 = range(len(degree_1))
x_2 = range(len(degree_2))
x_3 = range(len(degree_3))
x_4 = range(len(degree_4))
x_5 = range(len(degree_5))

y_1 = [z / float(sum(degree_1)) for z in degree_1]
y_2 = [z / float(sum(degree_2)) for z in degree_2]
y_3 = [z / float(sum(degree_3)) for z in degree_3]
y_4 = [z / float(sum(degree_4)) for z in degree_4]
y_5 = [z / float(sum(degree_5)) for z in degree_5]

fig =  plt.figure(num=1, figsize=(8, 5))
plt.loglog(x_1,y_1,color='red',linewidth=2) 
plt.plot(x_1,y_1,color='red',linewidth=1) 


plt.loglog(x_2,y_2,color='tomato',linewidth=2, linestyle = ':') 
plt.loglog(x_3,y_3,color='yellow',linewidth=2, linestyle = ':') 
plt.loglog(x_4,y_4,color='darkturquoise',linewidth=2, linestyle = ':') 
plt.loglog(x_5,y_5,color='fuchsia',linewidth=2, linestyle = ':') 
plt.xlabel('degree')
plt.ylabel('frequency')
plt.legend(loc=0, ncol=1)

plt.show() 
fig.savefig("ori_degree.svg", bbox_inches = 'tight')


ave_degree = G.degree()
ave_cluster = nx.average_clustering(G)
ave_path = nx.average_shortest_path_length(G)
sup_infpath = nx.diameter(G)

res = {'ave_degree': ave_degree,
       'ave_cluster': ave_cluster,
       'ave_path': ave_path,
       'sup_infpath': sup_infpath}


###############目标网络的统计###############
G1=nx.Graph()
G1.add_nodes_from(np.arange(1,max(data_a1['friendID'])))#加列表中的点
G1.add_weighted_edges_from(np.array(data_a1))

max_value = 0
for u in set(data_a1['userID']):
    temp = G1.degree(u)
    if (temp>max_value)&(temp<518):
        max_value = temp
        max_id = u
   
#max_id= 162;max_degree=534
#max_id = 4930;max_degree = 518
#max_id = 218; max_degree = 409

data_a2_1 = np.mean(data_a2['link'][np.isin(data_a2['userID'],162)] )
data_a2_2 = np.mean(data_a2['link'][np.isin(data_a2['userID'],4930)] )
data_a2_3 = np.mean(data_a2['link'][np.isin(data_a2['userID'],218)] )

data_a3_1 = np.mean(data_a3['link'][np.isin(data_a3['userID'],162)] )
data_a3_2 = np.mean(data_a3['link'][np.isin(data_a3['userID'],4930)] )
data_a3_3 = np.mean(data_a3['link'][np.isin(data_a3['userID'],218)] )

data_a4_1 = np.mean(data_a4['link'][np.isin(data_a4['userID'],162)] )
data_a4_2 = np.mean(data_a4['link'][np.isin(data_a4['userID'],4930)] )
data_a4_3 = np.mean(data_a4['link'][np.isin(data_a4['userID'],218)] )

data_a5_1 = np.mean(data_a5['link'][np.isin(data_a5['userID'],162)] )
data_a5_2 = np.mean(data_a5['link'][np.isin(data_a5['userID'],4930)] )
data_a5_3 = np.mean(data_a5['link'][np.isin(data_a5['userID'],218)] )

max_value = max(data_a1['userID'])
user = []
for i in range(max(data_a1['userID'])):
    temp = np.sum(np.isin(data_a1['userID'],i))
    user.append(temp)
    
per_user = np.percentile(np.array(user)[np.array(user)!=0],[25,50,75,100])
#per_user=1, 3, 7, 523
user = np.array(user)
per_3 = np.sum(user>5)/len(user)
per_2 = np.sum((user>1)&(user<=5))/len(user)
per_1 = np.sum((user>0)&(user<=1))/len(user)
per_0 = np.sum((user<=0))/len(user)

#0.049;0.05;0.355;0.538
index = np.array(range(len(user)))
user_cluster={}
user_cluster[3] = index[user>5]
user_cluster[2] = index[(user>1)&(user<=5)]
user_cluster[1] = index[(user>0)&(user<=1)]
user_cluster[0] = index[user<=0]

data_a2_3 = np.mean(data_a2['link'][np.isin(data_a2['userID'],user_cluster[3])] )
data_a2_2 = np.mean(data_a2['link'][np.isin(data_a2['userID'],user_cluster[2])] )
data_a2_1 = np.mean(data_a2['link'][np.isin(data_a2['userID'],user_cluster[1])] )
data_a2_0 = np.mean(data_a2['link'][np.isin(data_a2['userID'],user_cluster[0])] )

data_a3_3 = np.mean(data_a3['link'][np.isin(data_a3['userID'],user_cluster[3])] )
data_a3_2 = np.mean(data_a3['link'][np.isin(data_a3['userID'],user_cluster[2])] )
data_a3_1 = np.mean(data_a3['link'][np.isin(data_a3['userID'],user_cluster[1])] )
data_a3_0 = np.mean(data_a3['link'][np.isin(data_a3['userID'],user_cluster[0])] )

data_a4_3 = np.mean(data_a4['link'][np.isin(data_a4['userID'],user_cluster[3])] )
data_a4_2 = np.mean(data_a4['link'][np.isin(data_a4['userID'],user_cluster[2])] )
data_a4_1 = np.mean(data_a4['link'][np.isin(data_a4['userID'],user_cluster[1])] )
data_a4_0 = np.mean(data_a4['link'][np.isin(data_a4['userID'],user_cluster[0])] )


data_a5_3 = np.mean(data_a5['link'][np.isin(data_a5['userID'],user_cluster[3])] )
data_a5_2 = np.mean(data_a5['link'][np.isin(data_a5['userID'],user_cluster[2])] )
data_a5_1 = np.mean(data_a5['link'][np.isin(data_a5['userID'],user_cluster[1])] )
data_a5_0 = np.mean(data_a5['link'][np.isin(data_a5['userID'],user_cluster[0])] )


