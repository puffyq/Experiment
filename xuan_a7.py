# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:56:19 2019

@author: puffy
"""


import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
import scipy.special as scipy
import igraph
from evaluate import top_k,auc,caculate_cn
from sklearn import linear_model
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
np.random.seed(1024)


class xuan:
    def __init__(self, K = 30, Lambda = 0.5, Eta = 10):
        self.data = None
        self.train = None
        self.test = None

        self.user_group = None
        self.item_group = None

        self.user_count = None
        self.item_count = None
        self.data_count = None

        self.igroup_count = None
        self.ugroup_count = None

        self.K_factors = K
        self.Lambda = Lambda
        self.Eta = Eta
        self.Learning_rate = None

        self.V1 = None
        self.V2 = None
        self.V3 = None
        self.V4 = None

    def accumarray(self, P,n):
#P1_bag = self.accumarray(P1, self.user_count)
#P_bag的key是用户，values是同一个用户的index；可能用这个index在P2中搜索友邻
        P_bag = {}

        for i in range(1, n + 1):
            P_bag[i] = []

        for i in range(len(P)):
            bag_list = P_bag[P[i]]
            bag_list.append(i + 1)
            P_bag[P[i]] = bag_list

        return P_bag

    def accumRatings(slef, P, n):
#nr_user = self.accumRatings( np.int64(self.train[0]-1), self.user_count)
#P是用户名向量；n是用户数
        P_bag = {}
        #key是train中用户名；value是与用户相互关注的朋友个数
        for i in range(len(P)):
            if P[i] not in P_bag:
                P_bag[P[i]] = 1
            else:
                P_bag[P[i]] = P_bag[P[i]] + 1
        P_bag_list = []
        #长度是n，每个用户的朋友数都被统计出来包括没有关注任何人的用户
        for i in range(1, n + 1):
            if i in P_bag:
                P_bag_list.append(P_bag[i])
            else:
                P_bag_list.append(0)

        return P_bag_list

    def myridge(self,y, x, relation):
        if len(y) == 0:
#            betahat = np.zeros(self.K_factors)
            betahat = relation
        else:
            X = np.mat(np.dot(np.transpose(x), x) + np.eye(np.shape(x)[1]) * (self.Lambda + self.Eta) )
            k = np.linalg.matrix_rank(X)

            if k < np.shape(X)[0]:
                betahat = np.dot(np.linalg.pinv(X), np.dot(np.transpose(x), y)) + np.dot(np.linalg.pinv(X),self.Eta*np.transpose(relation))
            else:
                betahat = np.dot(X.I, np.dot(np.transpose(x), y)) + np.dot(X.I, self.Eta*np.transpose(relation))

        return betahat

    def regression(self,adj_matrix, y, x, r, flag):
        #betahat = self.regression(adj_q, V2_1, ps, link, 'q')
        if (flag == 'p')|(flag == 's'):
            X = np.mat(np.dot(np.transpose(x), x) + np.eye(np.shape(x)[1]) * (self.Lambda + self.Eta)) 
            k = np.linalg.matrix_rank(X)

            if k < np.shape(X)[0]:
                betahat = np.dot(np.dot(np.linalg.pinv(X), (np.dot(np.transpose(y), adj_matrix) + np.dot(np.dot(np.transpose(x),np.transpose(r)),adj_matrix))),np.linalg.pinv(np.dot(np.transpose(adj_matrix),adj_matrix))) 
            else:
                betahat = np.dot(np.dot(X.I, (self.Eta*np.dot(np.transpose(y), adj_matrix) + np.dot(np.dot(np.transpose(x),np.transpose(r)),adj_matrix))), np.linalg.pinv(np.dot(np.transpose(adj_matrix),adj_matrix)))
        else:
            X = np.mat(np.dot(np.transpose(x), x) + np.eye(np.shape(x)[1]) * (self.Lambda + self.Eta)) 
            k = np.linalg.matrix_rank(X)

            if k < np.shape(X)[0]:
                betahat = np.dot(np.dot(np.linalg.pinv(X), (np.dot(np.transpose(y), adj_matrix) + np.dot(np.dot(np.transpose(x),r),adj_matrix))),np.linalg.pinv(np.dot(np.transpose(adj_matrix),adj_matrix))) 
            else:
                betahat = np.dot(np.dot(X.I, (self.Eta*np.dot(np.transpose(y), adj_matrix) + np.dot(np.dot(np.transpose(x),r),adj_matrix))), np.linalg.pinv(np.dot(np.transpose(adj_matrix),adj_matrix)))
 
        return np.transpose(betahat)

    def grid_decent(self,y, x, old_):
        change = 999
        temp = old_.copy()
        if len(y) == 0:
            betahat = np.zeros(self.K_factors)
        else:
            betahat = temp.copy()
            y = np.array(y, dtype="float64")
            while (change > 1e-5):
                err_ = y - np.dot(x, temp)
                betahat += self.Learning_rate * (np.dot(err_, x) - self.Lambda * temp) / len(x)
                change = np.sum(np.square(betahat - temp))
                temp = betahat.copy()
        return betahat

    def stochastic_grid_decent(self,y, x, old_):
        change = 999
        temp = old_.copy()
        if len(y) == 0:
            betahat = np.zeros(self.K_factors)
            return betahat
        else:
            betahat = temp.copy()
            y = np.array(y, dtype="float64")
            while (True):
                ind = list(range(len(x)))
                random.shuffle(ind)
                for i in ind:
                    err_ = y[i] - np.dot(x[i], temp)
                    betahat += self.Learning_rate * (np.dot(err_, x[i]) - self.Lambda * temp)
                    change = np.sum(np.square(betahat - temp))
                    temp = betahat.copy()
                    if change <= 1e-5:
                        return betahat

    def ori_group(self, m1, m2):
# self.ori_group(clusters_num, clusters_num)
        self.ugroup_count = m1
        self.igroup_count = m2

        nr_user = self.accumRatings( np.int64(self.train[0]-1), self.user_count)
        nr_item = self.accumRatings( np.int64(self.train[1]-1), self.item_count)
#nr里面长度与用户数和朋友数相等，存储每个用户的友邻数
        qt_user = np.percentile(nr_user, np.linspace(0,m1,m1+1)*100/m1 )
        qt_item = np.percentile(nr_item, np.linspace(0,m2,m2+1)*100/m2 )
#qt存储朋友数的百分位,用来分组
        user_group = np.zeros(self.user_count)
        item_group = np.zeros(self.item_count)
#group里面要存储每个样本对中用户和朋友的分组信息
        print('users')
        for i in range(m1):
            user_group = user_group + i * ((nr_user>=qt_user[i]) & (nr_user<qt_user[i+1]))

        user_group = user_group + (m1-1) * (nr_user==qt_user[m1])
        user_group = user_group + 1
        self.user_group = user_group.astype(int)

        for j in range(m2):
            item_group = item_group + j * ((nr_item>=qt_item[j]) & (nr_item<qt_item[j+1]))

        item_group = item_group + (m2-1) * (nr_item==qt_item[m2])
        item_group = item_group + 1
        self.item_group = item_group.astype(int)

    def commu_group(self):
        U = np.zeros(shape=(self.user_count, self.item_count))

        for index, row in self.train.iterrows():
            u = int(row[0]) - 1
            i = int(row[1]) - 1
            r = int(row[2])
            U[u, i] = r

        # GET U CLUSTER
        print(U.shape)

        svd = TruncatedSVD(n_components=50, n_iter=5)
        U_dis = svd.fit_transform(U)
        h = np.corrcoef(U_dis)
        h[np.isnan(h)] = 0

        list_edge = []
        for i in range(1, len(h)):
            for k in range(1, len(h)):
                if h[i][k] >= 0.75 or h[i][k] <= -0.2:
                    list_edge.append((str(i), str(k), h[i][k]))
        print("USER NUM", len(h))
        print("EDGE NUM", len(list_edge))
        g = igraph.Graph.TupleList(list_edge, directed=False, vertex_name_attr='name', edge_attrs=None, weights=True)
        list_edge = []
        clusters = g.clusters().giant().community_spinglass()

        Ucluster = []
        for i in clusters.subgraphs():
            Ucluster.append([int(x) for x in i.vs["name"]])

        tri_clusters = []
        for i in g.clusters().subgraphs():
            tri_clusters.append([int(x) for x in i.vs["name"]])
        lenth = [len(x) for x in tri_clusters]
        del tri_clusters[lenth.index(max(lenth))]
        for i in tri_clusters:
            Ucluster.append(i)
        print("lenth",[len(i) for i in Ucluster])
        print("lenu", len(Ucluster))
        #
        # GET I CLUSTER
        svd = TruncatedSVD(n_components=50, n_iter=5)
        I_dis = svd.fit_transform(U.T)
        h = np.corrcoef(I_dis)

        h[np.isnan(h)] = 0

        list_edge = []
        for i in range(1, len(h)):
            for k in range(1, len(h)):
                if h[i][k] >= 0.8 or h[i][k] <= -0.2:
                    list_edge.append((str(i), str(k), h[i][k]))
        print("ITEM NUM", len(h))
        print("EDGE NUM", len(list_edge))
        g = igraph.Graph.TupleList(list_edge, directed=False, vertex_name_attr='name', edge_attrs=None, weights=True)

        list_edge = []
        clusters = g.clusters().giant().community_spinglass()

        Icluster = []
        for i in clusters.subgraphs():
            Icluster.append([int(x) for x in i.vs["name"]])

        tri_clusters = []
        for i in g.clusters().subgraphs():
            tri_clusters.append([int(x) for x in i.vs["name"]])
        lenth = [len(x) for x in tri_clusters]
        del tri_clusters[lenth.index(max(lenth))]
        for i in tri_clusters:
            Icluster.append(i)
        print("lenth", [len(i) for i in Icluster])
        print("leni", len(Icluster))

        user_group = np.zeros(self.user_count)
        item_group = np.zeros(self.item_count)

        for cluster_id in range(len(Ucluster)):
            for u in Ucluster[cluster_id]:
                user_group[u] = cluster_id
        for cluster_id in range(len(Icluster)):
            for i in Icluster[cluster_id]:
                item_group[i] = cluster_id
        user_group = user_group + 1
        item_group = item_group + 1
        

        self.user_group = user_group
        self.item_group = item_group
        self.ugroup_count = len(Ucluster)
        self.igroup_count = len(Icluster)

    def outer_group(self, ugroup, igroup):
        self.user_group = ugroup
        self.item_group = igroup
        self.ugroup_count = len(ugroup)
        self.igroup_count = len(igroup)

    def co_group(self, m1, m2, ufile, ifile):
        ufeature = pd.read_csv(ufile)

        feature_u = [name for name in list(ufeature.columns) if name != 'index']
        
        kmeans_u = KMeans(n_clusters=m1)
        kmeans_u.fit(ufeature[feature_u])
    
        u_group_hash = {}
        index = 0
        for group in kmeans_u.labels_:
            if group not in u_group_hash:
                u_group_hash[group] = []
            u_group_hash[group].append(index)
            index += 1

        u_group_list = list(u_group_hash.values())

        ifeature = pd.read_csv(ifile)

        feature_i = [name for name in list(ifeature.columns) if name != 'index']

        kmeans_i = KMeans(n_clusters=m2)
        kmeans_i.fit(ifeature[feature_i])

        i_group_hash = {}
        index = 0
        for group in kmeans_i.labels_:
            if group not in i_group_hash:
                i_group_hash[group] = []
            i_group_hash[group].append(index)
            index += 1
        i_group_list = list(i_group_hash.values())
        
        user_group = np.zeros(self.user_count)
        item_group = np.zeros(self.item_count)

        for cluster_id in range(len(u_group_list)):
            for u in u_group_list[cluster_id]:
                if u < self.user_count:
                    user_group[u] = cluster_id
        for cluster_id in range(len(i_group_list)):
            for i in i_group_list[cluster_id]:
                if i < self.item_count:
                    item_group[i] = cluster_id
        self.user_group = int(user_group+1)
        self.item_group = int(item_group+1)
        self.ugroup_count = m1
        self.igroup_count = m2

    def iterate(self,length, number, iterate, adj_matrix, train, test, max_iter, lr=0.05, alg='LSE', cluster='ORI', clusters_num=10, ufile=None, ifile=None):
 
 #加入组间效应的SVD
 #train和test都是pandas.dataframe类型的
        train[[0,1]] = train[[0,1]].astype(int)
        test[[0,1]] = test[[0,1]].astype(int)
        self.train = train
        self.test = test
        self.data = pd.concat([train, test])


        self.user_count = number
        self.item_count = number
        self.data_count = len(self.data)
        self.Learning_rate = lr
#获取数据的信息，包括用户数和朋友数，相互关注的样本数，学习率
        print (self.user_count)
        print('Building the cluster...')
        if cluster == 'ORI':
#使用朋友个数信息进行聚类
            self.ori_group(clusters_num, clusters_num)
        elif cluster == 'CN':
            self.commu_group()
        elif cluster == 'CO':
            self.co_group(clusters_num, clusters_num, ufile=ufile, ifile=ifile)
        else:
            print("UNDEFINED CLUSTERING ALGRITHM:", cluster)
#初始化四个矩阵，V1=P,V2=Q,V3=S,V4=T           
        self.V1 = np.array(np.random.normal(loc=0, scale=0.3, size=(self.user_count, self.K_factors)), dtype='float64')
        self.V2 = np.array(np.random.normal(loc=0, scale=0.3, size=(self.item_count, self.K_factors)), dtype='float64')
        self.V3 = np.array(np.random.normal(loc=0, scale=0.3, size=(self.ugroup_count, self.K_factors)), dtype='float64')
        self.V4 = np.array(np.random.normal(loc=0, scale=0.3, size=(self.igroup_count, self.K_factors)), dtype='float64')

        print('Pre-calculating the index...')
        P1 = np.int64(train[0])
        P2 = np.int64(train[1])

        print('P1')
        P1_bag = self.accumarray(P1, self.user_count)
        print('P2')
        P2_bag = self.accumarray(P2, self.item_count)
#G1用户的分组；G2友邻的分组
        print (self.user_group[train[0].astype(int) - 1])
        G1 = np.int64(self.user_group[train[0] - 1])
        G2 = np.int64(self.item_group[train[1] - 1])
#G1_bag里放分组下的用户名；G2_bag里放分组下的友邻名
        print('G1')
        G1_bag = self.accumarray(G1, self.ugroup_count)
        print('G2')
        G2_bag = self.accumarray(G2, self.igroup_count)

        print('.complete.')

        y1_bag = {}
        y2_bag = {}
        y3_bag = {}
        y4_bag = {}
                
        x1_bag = {}
        x2_bag = {}
        x3_bag = {}
        x4_bag = {}
        
#之后V1,2,3,4没有参与运算，他们的初始化的职能由V1_init,V2_init,V3_init,V4_init代替
#U1,U2,U3,U4_init作为整体（ALS）来更新参与loss的计算；V1,V2,V3,V4_init作为ALS中某一步的backfitting来更新参与loss计算
        U1_init = self.V1.copy()
        U2_init = self.V2.copy()
        U3_init = self.V3.copy()
        U4_init = self.V4.copy()

        V1_init = U1_init.copy()
        V2_init = U2_init.copy()
        V3_init = U3_init.copy()
        V4_init = U4_init.copy()
        
        link = np.zeros((self.user_count,self.item_count))
        index = (np.array((train.values[:,0]-1).astype(int)),np.array((train.values[:,1]-1).astype(int)))
        link[index] = train.values[:,2]

        iter_ = 1
        diff_all = 1
        beta_maxit = 15

        while diff_all > 1e-3:
#最外层循环控制整体的损失
            print('-Iter' + str(iter_))

            # matlab: V1,V2,V3,V4
            V1_1 = U1_init.copy()
            V2_1 = U2_init.copy()
            V3_1 = U3_init.copy()
            V4_1 = U4_init.copy()

            diff_item = 1
            

            print('Updating items...')
            X_user = V1_1[train[0] - 1] + V3_1[G1 - 1]
#X_USER = (P+S)
            beta_itra = 0            
            while diff_item > 1e-5:
            #控制ALS一次迭代中backfitting的损失
                y = train[2] - np.sum(np.multiply(X_user, V4_1[G2 - 1]), axis=1)
                #(P+S)'(T);更新Q即友邻的效应
                #对Q进行建模和估计，Q是V2_init
                tc = np.int64(self.item_group[np.arange(self.item_count)])
                sc = np.int64(self.user_group[np.arange(self.user_count)])
                ps = V1_1 + V3_1[sc-1]
                beta_init = np.array(np.random.normal(loc=0, scale=0.3, size=(12, self.K_factors)), dtype='float64')
                beta_diff = 1

                while beta_diff > 1e-5:
                    #    def regression(self,adj_matrix,y, x, r):
                    beta_loss_q = sum(np.square(train[2] - np.sum(np.multiply(X_user, (V2_1[train[1] - 1] + V4_1[G2 - 1])), axis=1)))/len(train[2])
                    adj_q = (np.c_[adj_matrix[str(iterate)+'_a2_q'], adj_matrix[str(iterate)+'_a3_q'], 
                                  adj_matrix[str(iterate)+'_a4_q'], adj_matrix[str(iterate)+'_a5_q']])                 
                    betahat = self.regression(adj_q, V2_1, ps, link, 'q')
                    q_hat = np.dot(adj_q,betahat)
                    q_hat = np.array(q_hat)
                    beta_diff = sum(sum(np.square(np.array(betahat) - np.array(beta_init)))) / adj_q.shape[1] / self.K_factors 
                    print('iter '+str(beta_itra))
                    print("Mean squared error: %.4f" % mean_squared_error(V2_1, q_hat))  
                    print('Variance score: %.4f' % r2_score(V2_1, q_hat))

                    V2_1 = q_hat
                    beta_init = betahat
                    
#                    print(beta_loss_)

                
                for i in range(1, self.item_count + 1):
                    if len(P2_bag[i]) > 0:
                    #如果这个友邻存在
                    #y2_bag放这个友邻的rank；x2_bag放这个友邻的（P+S）效应
                        y2_bag[i] = y[np.array(P2_bag[i]) - 1]
                        x2_bag[i] = X_user[np.array(P2_bag[i]) - 1]
                    else:
                        y2_bag[i] = []
                        x2_bag[i] = []
                       
                for i in range(1, self.item_count + 1):
                    if alg == 'LSE':                        
                        V2_1[i - 1] = self.myridge(y2_bag[i], x2_bag[i], q_hat[i-1])
                    elif alg == 'GD':
                        V2_1[i-1] = self.grid_decent(y2_bag[i], x2_bag[i], V2_1[i-1])
                    elif alg == 'SGD':
                        V2_1[i-1] = self.stochastic_grid_decent(y2_bag[i], x2_bag[i], V2_1[i-1])
                    else:
                        print("UNDEFINED ITERATE ALGRITHM")

                y = train[2] - np.sum(np.multiply(X_user, V2_1[train[1] - 1]), axis=1)
#更新Q只需要使（P+S）Q的损失最小，因为（P+S）(T)这一项相当于常数
                #对T进行建模和估计，T是V4_1
                sc = np.int64(self.user_group[np.arange(self.user_count)])
                tc = np.int64(self.item_group[np.arange(self.item_count)])
                ps = V1_1 + V3_1[sc-1]
                beta_init = np.array(np.random.normal(loc=0, scale=0.3, size=(12, self.K_factors)), dtype='float64')
                beta_diff = 1

                while beta_diff > 1e-5:
                    #    def regression(self,adj_matrix,y, x, r):
                    beta_loss_t = sum(np.square(train[2] - np.sum(np.multiply(X_user, (V2_1[train[1] - 1] + V4_1[G2 - 1])), axis=1)))/len(train[2])
                    adj_t = (np.c_[adj_matrix[str(iterate)+'_a2_t'], adj_matrix[str(iterate)+'_a3_t'], 
                                  adj_matrix[str(iterate)+'_a4_t'], adj_matrix[str(iterate)+'_a5_t']])
                    betahat = self.regression(adj_t[tc-1], V4_1[tc-1], ps, link, 't')
                    t_hat = np.dot(adj_t,betahat)
                    t_hat = np.array(t_hat)
                    beta_diff = sum(sum(np.square(np.array(betahat) - np.array(beta_init)))) / adj_t.shape[1] / self.K_factors 
                    print('iter '+str(beta_itra))
                    print("Mean squared error: %.4f" % mean_squared_error(V4_1, t_hat))  
                    print('Variance score: %.4f' % r2_score(V4_1, t_hat))
                    V4_1 = t_hat
                    beta_init = betahat
                
#                print(beta_loss)
                                    
                for i in range(1, self.igroup_count + 1):
                #更新T
                #x4_bag里放这一组用户的X_USER效应
                    if len(G2_bag[i]) > 0:
                        y4_bag[i] = y[np.array(G2_bag[i]) - 1]
                        x4_bag[i] = X_user[np.array(G2_bag[i]) - 1]
                    else:
                        y4_bag[i] = []
                        x4_bag[i] = []
                             
                for i in range(1, self.igroup_count + 1):
                    if alg == 'LSE':
                        V4_1[i - 1] = self.myridge(y4_bag[i], x4_bag[i], t_hat[i-1])
                    elif alg == 'GD':
                        V4_1[i-1] = self.grid_decent(y4_bag[i], x4_bag[i], V4_1[i-1])
                    elif alg == 'SGD':
                        V4_1[i-1] = self.stochastic_grid_decent(y4_bag[i], x4_bag[i], V4_1[i-1])
                    else:
                        print("UNDEFINED ITERATE ALGRITHM")

                diff_item = sum(sum(np.square(V2_1 - V2_init))) / self.item_count / self.K_factors + sum(
                    sum(np.square(V4_1 - V4_init))) / self.igroup_count / self.K_factors
#这个diff的要求不在于控制loss，而在于迭代稳定了，不太变化了，信息提取全面了
                V2_init = V2_1.copy()
                V4_init = V4_1.copy()
                
                beta_itra = beta_itra + 1 
                if beta_itra > beta_maxit:
                    break
#                print(beta_loss)                

            print('Done')

            diff_user = 1

            print('Updating users...')
            X_item = V2_1[train[1] - 1] + V4_1[G2 - 1]

            beta_itra = 0
            while diff_user > 1e-5:
            #控制ALS一次迭代中backfitting的损失
                y = train[2] - np.sum(np.multiply(X_item, V3_1[G1 - 1]), axis=1)
                #对P进行建模和估计，P是V1_1
                tc = np.int64(self.item_group[np.arange(self.item_count)])
                qt = V2_1 + V4_1[tc-1]
                beta_init = np.array(np.random.normal(loc=0, scale=0.3, size=(12, self.K_factors)), dtype='float64')
                beta_diff = 1
                
                while beta_diff > 1e-5:
                    #    def regression(self,adj_matrix,y, x, r):
                    beta_loss_p = sum(np.square(train[2] - np.sum(np.multiply(V1_1[train[0] - 1] + V3_1[G1 - 1], X_item), axis=1)))/len(train[2])
                    adj_p = (np.c_[adj_matrix[str(iterate)+'_a2_p'], adj_matrix[str(iterate)+'_a3_p'], 
                                  adj_matrix[str(iterate)+'_a4_p'], adj_matrix[str(iterate)+'_a5_p']])
                    betahat = self.regression(adj_p, V1_1, qt, link, 'p')
                    p_hat = np.dot(adj_p,betahat)
                    p_hat = np.array(p_hat)
                    beta_diff = sum(sum(np.square(np.array(betahat) - np.array(beta_init)))) / adj_p.shape[1] / self.K_factors 
                    print('iter '+str(beta_itra))
                    print("Mean squared error: %.4f" % mean_squared_error(V1_1, p_hat))  
                    print('Variance score: %.4f' % r2_score(V1_1, p_hat))
                    V1_1 = p_hat
                    beta_init = betahat 
                  
#                print(beta_loss)                
                
                for i in range(1, self.user_count + 1):
                    if len(P1_bag[i]) > 0:
                        y1_bag[i] = y[np.array(P1_bag[i]) - 1]
                        x1_bag[i] = X_item[np.array(P1_bag[i]) - 1]
                    else:
                        y1_bag[i] = []
                        x1_bag[i] = []
    
                for i in range(1, self.user_count + 1):
                    if alg == 'LSE':
                        V1_1[i - 1] = self.myridge(y1_bag[i], x1_bag[i], p_hat[i-1])
                    elif alg == 'GD':
                        V1_1[i-1] = self.grid_decent(y1_bag[i], x1_bag[i], V1_1[i-1])
                    elif alg == 'SGD':
                        V1_1[i-1] = self.stochastic_grid_decent(y1_bag[i], x1_bag[i], V1_1[i-1])
                    else:
                        print("UNDEFINED ITERATE ALGRITHM")


                y = train[2] - np.sum(np.multiply(X_item, V1_1[train[0] - 1]), axis=1)

                #对S进行建模和估计，S是V3_1
                sc = np.int64(self.user_group[np.arange(self.user_count)])
                tc = np.int64(self.item_group[np.arange(self.item_count)])
                qt = V2_1 + V4_1[tc-1]
                beta_init = np.array(np.random.normal(loc=0, scale=0.3, size=(12, self.K_factors)), dtype='float64')
                beta_diff = 1

                while beta_diff > 1e-5:
                    #    def regression(self,adj_matrix,y, x, r):
                    beta_loss_s = sum(np.square(train[2] - np.sum(np.multiply((V1_1[train[0] - 1] + V3_1[G1 - 1]), X_item), axis=1)))/len(train[2])
                    adj_s = (np.c_[adj_matrix[str(iterate)+'_a2_s'], adj_matrix[str(iterate)+'_a3_s'], 
                                  adj_matrix[str(iterate)+'_a4_s'], adj_matrix[str(iterate)+'_a5_s']])
                    betahat = self.regression(adj_s[sc-1], V3_1[sc-1], qt, link, 's')
                    s_hat = np.dot(adj_s,betahat)
                    s_hat = np.array(s_hat)
                    beta_diff = sum(sum(np.square(np.array(betahat) - np.array(beta_init)))) / adj_s.shape[1] / self.K_factors 
                    print('iter '+str(beta_itra))
                    print("Mean squared error: %.4f" % mean_squared_error(V3_1, s_hat))  
                    print('Variance score: %.4f' % r2_score(V3_1, s_hat))
                    V3_1 = s_hat
                    beta_init = betahat
                
#                print(beta_loss)

                
                for i in range(1, self.ugroup_count + 1):

                    if len(G1_bag[i]) > 0:
                        y3_bag[i] = y[np.array(G1_bag[i]) - 1]
                        x3_bag[i] = X_item[np.array(G1_bag[i]) - 1]
                    else:
                        y3_bag[i] = []
                        x3_bag[i] = []
     
                for i in range(1, self.ugroup_count + 1):
                    if alg == 'LSE':
                        V3_1[i - 1] = self.myridge(y3_bag[i], x3_bag[i], s_hat[i-1])
                    elif alg == 'GD':
                        V3_1[i-1] = self.grid_decent(y3_bag[i], x3_bag[i], V3_1[i-1])
                    elif alg == 'SGD':
                        V3_1[i-1] = self.stochastic_grid_decent(y3_bag[i], x3_bag[i], V3_1[i-1])
                    else:
                        print("UNDEFINED ITERATE ALGRITHM")
                    
                diff_user = sum(sum(np.square(V1_1 - V1_init))) / self.user_count / self.K_factors + sum(
                    sum(np.square(V3_1 - V3_init))) / self.ugroup_count / self.K_factors

                V1_init = V1_1.copy()
                V3_init = V3_1.copy()
                
                beta_itra = beta_itra + 1 
                if beta_itra > beta_maxit:
                    break  

            print('Done')

            diff_all = sum(sum(np.square(U1_init + U3_init[np.int64(self.user_group) - 1] \
                                         - V1_init - V3_init[np.int64(self.user_group) - 1]))) / self.user_count / self.K_factors \
                       + sum(sum(np.square(U2_init + U4_init[np.int64(self.item_group) - 1] \
                                           - V2_init - V4_init[np.int64(self.item_group) - 1]))) / self.item_count / self.K_factors

            print('Improvement is ' + str(diff_all) + '.')

            U1_init = V1_init.copy()
            U2_init = V2_init.copy()
            U3_init = V3_init.copy()
            U4_init = V4_init.copy()

            iter_ = iter_ + 1

            if iter_ > max_iter:
                print('Not converged; maximum number of iterations achieved!\n')
                break

#        rmse_te = np.sqrt(np.mean(np.square(test[2] -
#                                            np.sum(np.multiply(U1_init[test[0] - 1] +  \
#                                                               U3_init[np.int64(self.user_group[test[0] - 1]) - 1],
#                                                               U2_init[test[1] - 1] +  \
#                                                               U4_init[np.int64(self.item_group[test[1] - 1]) - 1]),
#                                                   axis=1))))
        print('beta_loss_q:'+str(beta_loss_q)+';beta_loss_t;'+str(beta_loss_t)+
              ';beta_loss_p:'+str(beta_loss_p)+';beta_loss_s:'+str(beta_loss_s))
            
        cn_all = caculate_cn(number, train)
        y_hat = np.sum(np.multiply(U1_init[test[0] - 1] +  \
                                   U3_init[np.int64(self.user_group[test[0] - 1]) - 1],
                                   U2_init[test[1] - 1] +  \
                                   U4_init[np.int64(self.item_group[test[1] - 1]) - 1]),
                     axis=1)
#        length = len(set(np.array(test[0])[~np.isin(test[2],0.5)]))
        recall,precision,recall_base,precision_base,f1,f1_base = top_k(self.data, test, cn_all, self.user_group, self.item_group, U1_init,U2_init,U3_init,U4_init,k = length )     #     k=int(0.3*number)                 
        auc_value = auc(test, y_hat)

        
        train_hat = np.sum(np.multiply(U1_init[train[0] - 1] +  \
                                   U3_init[np.int64(self.user_group[train[0] - 1]) - 1],
                                   U2_init[train[1] - 1] +  \
                                   U4_init[np.int64(self.item_group[train[1] - 1]) - 1]),
                     axis=1)
#        length_train = len(set(np.array(train[0])[~np.isin(train[2],0.5)]))
        train_recall,train_precision,train_recall_base,train_precision_base,train_f1,train_f1_base = top_k(self.data, train, cn_all, self.user_group, self.item_group, U1_init,U2_init,U3_init,U4_init,k = length )     #     k=int(0.3*number)                 
        train_auc_value = auc(train, train_hat)
        
        
        evaluate_matrix={'auc_value':auc_value,'precision':precision,'recall':recall, 'f1':f1}
        evaluate_train_matrix={'auc_value':train_auc_value,'precision':train_precision,'recall':train_recall, 'f1':train_f1}
        
        base_matrix = {'precision_base':precision_base, 'recall_base':recall_base, 'f1_base':f1_base}
        
        effective_matrix = []
        effective_matrix.append(U1_init)
        effective_matrix.append(U2_init)
        effective_matrix.append(U3_init)
        effective_matrix.append(U4_init)    
        effective_matrix.append(length)
        effective_matrix.append(evaluate_train_matrix)
        effective_matrix.append(evaluate_matrix)   
        effective_matrix.append(base_matrix)
#        
#        np.save("U1_init",U1_init)
#        np.save("U2_init",U2_init)
#        np.save("U3_init",U3_init)
#        np.save("U4_init",U4_init)

        return effective_matrix
    
    
    
    
    
    
    