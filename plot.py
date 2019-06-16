# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:58:15 2019

@author: puffy
"""

import matplotlib.pyplot as plt
import numpy as np

x = [50, 100, 150, 200, 250]

precision_fuse = [0.032, 0.027, 0.02, 0.018, 0.022]
precision_gssvd = [0.015, 0.014, 0.014, 0.012, 0.019]
precision_salton = [0.023, 0.0175, 0.014, 0.01, 0.009]

recall_fuse = [0.18, 0.316, 0.35, 0.42, 0.57]
recall_gssvd = [0.09, 0.08, 0.09, 0.087, 0.13]
recall_salton = [0.13, 0.14, 0.2, 0.21, 0.22]

fig = plt.figure(num=2, figsize=(6, 3))

plt.subplot(1,2,1)
plt.plot(x, precision_fuse,'-+', color='red', linewidth=1.5, markersize=3,markeredgewidth=3, label = 'fuse')
plt.plot(x, precision_gssvd,'-o', color = 'tomato', linewidth = 1.5,linestyle = ':', markersize=3,markeredgewidth=3, label ='GSSVD')
plt.plot(x, precision_salton, '-h', color = 'yellow',linewidth = 1.5,linestyle = ':', markersize=3,markeredgewidth=3, label = 'salton')

plt.xlabel('K')
plt.ylabel('precision')
#plt.legend(loc=0, ncol=2)
plt.xticks(x)

plt.subplot(1,2,2)
plt.plot(x, recall_fuse,'-+', color='red', linewidth=1.5, markersize=3, markeredgewidth=3, label = 'fuse')
plt.plot(x, recall_gssvd,'-o', color = 'tomato', linewidth = 1.5, linestyle = ':', markersize=3,markeredgewidth=3, label ='GSSVD')
plt.plot(x, recall_salton, '-h', color = 'yellow',linewidth = 1.5, linestyle = ':', markersize=3,markeredgewidth=3, label = 'salton')

plt.xlabel('K')
plt.ylabel('recall')
plt.legend(loc=0, ncol=1)

plt.xticks(x)
plt.subplots_adjust(wspace =0.4)
plt.show()

fig.savefig('top_k.svg', bbox_inches = 'tight')#, 