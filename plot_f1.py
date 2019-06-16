# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:47:15 2019

@author: puffy
"""

import matplotlib.pyplot as plt
import numpy as np
import os

#推荐长度为50
x = [50, 100, 150, 200, 250]#28,
f1_fuse = [0.083, 0.0578, 0.049, 0.044, 0.039]#0.0857, 
f1_ggsvd = [0.054, 0.0375, 0.034, 0.034, 0.032]#0.0839, 
f1_salton = [0.038, 0.032, 0.025, 0.02, 0.016]#0.028,

precision_fuse = [0.0486, 0.0315, 0.026, 0.023, 0.0205]#0.047, 
precision_ggsvd = [0.0326, 0.0205, 0.018, 0.017, 0.016]#0.046, 
precision_salton = [0.0235, 0.018, 0.013, 0.011, 0.008]#0.016,

recall_fuse = [0.374, 0.39, 0.49, 0.55, 0.6075]#0.546,
recall_ggsvd = [0.288, 0.29, 0.385, 0.46, 0.38]#0.52, 
recall_salton = [0.126, 0.168, 0.186, 0.19, 0.1975]#0.126, 


fig = plt.figure(num=3, figsize=(15, 5))

plt.subplot(1,3,1)
plt.plot(x, precision_fuse,'-+', color='red', linewidth=1.5, markersize=3,markeredgewidth=3, label = 'fuse')
plt.plot(x, precision_ggsvd,'-o', color = 'tomato', linewidth = 1.5,linestyle = ':', markersize=3,markeredgewidth=3, label ='GSSVD')
plt.plot(x, precision_salton, '-h', color = 'yellow',linewidth = 1.5,linestyle = ':', markersize=3,markeredgewidth=3, label = 'salton')

plt.xlabel('K')
plt.ylabel('precision')
plt.xticks(x)
#plt.legend(loc=0, ncol=2)

plt.subplot(1,3,2)
plt.plot(x, recall_fuse,'-+', color='red', linewidth=1.5, markersize=3, markeredgewidth=3, label = 'fuse')
plt.plot(x, recall_ggsvd,'-o', color = 'tomato', linewidth = 1.5, linestyle = ':', markersize=3,markeredgewidth=3, label ='GSSVD')
plt.plot(x, recall_salton, '-h', color = 'yellow',linewidth = 1.5, linestyle = ':', markersize=3,markeredgewidth=3, label = 'salton')

plt.xlabel('K')
plt.ylabel('recall')
plt.xticks(x)

plt.subplot(1,3,3)
plt.plot(x, f1_fuse,'-+', color='red', linewidth=1.5, markersize=3, markeredgewidth=3, label = 'fuse')
plt.plot(x, f1_ggsvd,'-o', color = 'tomato', linewidth = 1.5, linestyle = ':', markersize=3,markeredgewidth=3, label ='GSSVD')
plt.plot(x, f1_salton, '-h', color = 'yellow',linewidth = 1.5, linestyle = ':', markersize=3,markeredgewidth=3, label = 'salton')

plt.xlabel('K')
plt.ylabel('F1')
plt.xticks(x)

plt.legend(loc=0, ncol=1)

plt.subplots_adjust(wspace =0.2)
plt.show()


outdir = r"C:\data\linkPre_v2\linkPre\data_sto\pic"
outpath = os.path.join(outdir,"f1_new.svg")

fig.savefig(outpath, bbox_inches = 'tight')