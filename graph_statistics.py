# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:45:26 2019

@author: puffy
"""
import os
import numpy as np
import networkx as nx

dir_path = r"C:\data\linkPre_v2\linkPre\data_sto\subgraph"

def do_process(number, iterate,flag):
    subDir_path = os.path.join(dir_path, str(number))
    name = str(iterate)+'_sub_a'+str(flag+1)+'_'+str(number)+'.npy'
    file_path = os.path.join(subDir_path, name)
    data = np.load(file_path)
    G=nx.Graph()
    G.add_nodes_from(np.arange(1,number))#加列表中的点
#    G.add_edges_from(data[:,[0,1]])
#    for i in aa:
#        l=nx.average_shortest_path_length(i)
#        print (l)
    G.add_weighted_edges_from(data)
    ave_path = 0
    for c in sorted(nx.connected_components(G),key=len,reverse=True):
        ave_path = ave_path + len(c)

    ave_path = ave_path/len(list(nx.connected_components(G)))        
    sup_infpath = len(max(nx.connected_components(G),key=len))
    sum_nodes = len(G.nodes())
    sum_edges = len(G.edges())
    ave_degree = np.mean(G.degree())
    ave_cluster = nx.average_clustering(G)
#    while True:
#        try:
#           ave_path = nx.average_shortest_path_length(G)
#           break
#        except nx.exception.NetworkXError:
#            ave_path = 0
#            break
#    while True:
#        try:
#            sup_infpath = nx.diameter(G)
#            break
#        except nx.exception.NetworkXError:
#            sup_infpath = 0
#            break

    res = {'ave_degree': ave_degree,
           'ave_cluster': ave_cluster,
           'ave_path': ave_path,
           'sup_infpath': sup_infpath,
           'sum_nodes':sum_nodes,
           'sum_edges':sum_edges}
        
    return res

if __name__ == '__main__':
    #输入subgraph的路径
    #输入number和iterate
    res_all = {}    
    number = 4000  
    inde = 10
    for i in range(5):

        ave_degree = 0
        ave_cluster = 0
        ave_path = 0
        sup_infpath = 0
        sum_nodes = 0
        sum_edges = 0         
        for j in range(inde):
            iterate = j+1
            res = do_process(number, iterate, i)        
            ave_degree =ave_degree + res['ave_degree']
            ave_cluster =ave_cluster + res['ave_cluster']
            ave_path = ave_path + res['ave_path']
            sup_infpath = sup_infpath + res['sup_infpath']
            sum_nodes = sum_nodes + res['sum_nodes']
            sum_edges = sum_edges + res['sum_edges']
            
        res_all['a'+str(i+1)]= [float(ave_degree)/inde,float(ave_cluster)/inde,
                       float(ave_path)/inde, float(sup_infpath)/inde, float(sum_nodes)/inde,
                       float(sum_edges)/inde]
            