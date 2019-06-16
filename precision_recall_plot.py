# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:47:15 2019

@author: puffy
"""
import os
import matplotlib.pyplot as plt
import numpy as np

#推荐长度为50
x = [100,500,1000,2000,3000]

F1_fuse = [0.15, 0.073, 0.08, 0.064, 0.048]
F1_ggsvd = [0.13, 0.056, 0.048, 0.027, 0.016]
F1_salton = [0.03, 0.032, 0.043, 0.048, 0.04]

precision_fuse = [0.084, 0.044, 0.047, 0.039, 0.029]
precision_ggsvd = [0.077, 0.033, 0.028, 0.016, 0.009]
precision_salton = [0.018, 0.021, 0.026, 0.029, 0.026]

recall_fuse = [0.89, 0.34, 0.29, 0.2, 0.15]
recall_ggsvd = [0.83, 0.29, 0.18, 0.09, 0.05]
recall_salton = [0.13, 0.09, 0.134, 0.15, 0.13]


fig = plt.figure(num=3, figsize=(15, 5))

plt.subplot(1,3,1)
plt.plot(x, precision_fuse,'-+', color='red', linewidth=1.5, markersize=3,markeredgewidth=3, label = 'fuse')
plt.plot(x, precision_ggsvd,'-o', color = 'tomato', linewidth = 1.5,linestyle = ':', markersize=3,markeredgewidth=3, label ='GSSVD')
plt.plot(x, precision_salton, '-h', color = 'yellow',linewidth = 1.5,linestyle = ':', markersize=3,markeredgewidth=3, label = 'salton')

plt.xlabel('degree')
plt.ylabel('precision')
plt.xticks(x)
#plt.legend(loc=0, ncol=2)

plt.subplot(1,3,2)
plt.plot(x, recall_fuse,'-+', color='red', linewidth=1.5, markersize=3, markeredgewidth=3, label = 'fuse')
plt.plot(x, recall_ggsvd,'-o', color = 'tomato', linewidth = 1.5, linestyle = ':', markersize=3,markeredgewidth=3, label ='GSSVD')
plt.plot(x, recall_salton, '-h', color = 'yellow',linewidth = 1.5, linestyle = ':', markersize=3,markeredgewidth=3, label = 'salton')

plt.xlabel('degree')
plt.ylabel('recall')
plt.xticks(x)

plt.subplot(1,3,3)
plt.plot(x, F1_fuse,'-+', color='red', linewidth=1.5, markersize=3, markeredgewidth=3, label = 'fuse')
plt.plot(x, F1_ggsvd,'-o', color = 'tomato', linewidth = 1.5, linestyle = ':', markersize=3,markeredgewidth=3, label ='GSSVD')
plt.plot(x, F1_salton, '-h', color = 'yellow',linewidth = 1.5, linestyle = ':', markersize=3,markeredgewidth=3, label = 'salton')

plt.xlabel('degree')
plt.ylabel('F1')
plt.xticks(x)


plt.legend(loc=0, ncol=1)

plt.subplots_adjust(wspace =0.2)
plt.show()

outdir = r"C:\data\linkPre_v2\linkPre\data_sto\pic"
outpath = os.path.join(outdir,"precision_recall_new.svg")

fig.savefig(outpath, bbox_inches = 'tight')