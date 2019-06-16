# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 19:19:40 2019

@author: puffy
"""

import matplotlib.pyplot as plt
import numpy as np
#平均路径ap
x = [100,500,1000,2000,3000]
a1_ap = [1.86, 5.73, 3.56, 4.38, 3.91]
a2_ap = [56.96, 39.57, 27.5, 14.75, 12.7]
a3_ap = [5.4, 13.5, 44.39, 18.36, 12.06]
a4_ap = [2.5, 5.07, 7.05, 7.83, 7.2]
a5_ap = [11.75, 5.25, 9.37, 9.56, 21.45]

a1_c = [0.03, 0.14, 0.09, 0.11, 0.09]
a2_c = [0.58, 0.65, 0.65, 0.51, 0.48]
a3_c = [0.56, 0.6, 0.7, 0.7, 0.65]
a4_c = [0.37, 0.54, 0.57, 0.57, 0.56]
a5_c = [0.63, 0.44, 0.56, 0.57, 0.6]

a1_ds = [39.3, 349.33, 650, 1515, 2206.33]
a2_ds = [95, 477.67, 957.33, 1864.4, 2761.33]
a3_ds = [80.67, 394.33, 935.33, 1824.3, 2739.33]
a4_ds = [61, 365, 796, 1651.2, 2411.67]
a5_ds = [92, 391.33, 889.33, 1775.67, 2781.33]

a1_de = [25.63, 130.8,253.83, 507.21, 756.74]
a2_de = [42.49, 201.48, 383.35, 589.06, 855.22]
a3_de = [35.75, 179.95, 396.75, 790.28,1054.67]
a4_de = [28.88, 173.29, 326.44, 635.34, 905.42]
a5_de = [39.22, 145.41, 314.4, 613.72, 986.5]



fig = plt.figure()
plt.subplot(2,2,1)#, constrained_layout=True
plt.plot(x,a1_ap,color='red',linewidth=2, label = 'target')
plt.plot(x,a2_ap,color='tomato',linewidth=2, linestyle = ':', label='friends')
plt.plot(x,a3_ap,color='yellow',linewidth=2, linestyle = ':', label = 'subscriptions')
plt.plot(x,a4_ap,color='darkturquoise',linewidth=2, linestyle = ':', label = 'subscribers')
plt.plot(x,a5_ap,color='fuchsia',linewidth=2, linestyle = ':', label = 'videos')
plt.xlabel('degree')
plt.ylabel('average path')
plt.text(0 , 1, u'a')


plt.subplot(2,2,2)#, constrained_layout=True
plt.plot(x,a1_c,color='red',linewidth=2, label = 'target')
plt.plot(x,a2_c,color='tomato',linewidth=2, linestyle = ':', label='friends')
plt.plot(x,a3_c,color='yellow',linewidth=2, linestyle = ':', label = 'subscriptions')
plt.plot(x,a4_c,color='darkturquoise',linewidth=2, linestyle = ':', label = 'subscribers')
plt.plot(x,a5_c,color='fuchsia',linewidth=2, linestyle = ':', label = 'videos')
plt.xlabel('degree')
plt.ylabel('average clustering coefficient')
plt.text(1,0 ,u'b')

plt.subplot(2,2,3)#, constrained_layout=True
plt.plot(x,a1_ds,color='red',linewidth=2, label = 'target')
plt.plot(x,a2_ds,color='tomato',linewidth=2, linestyle = ':', label='friends')
plt.plot(x,a3_ds,color='yellow',linewidth=2, linestyle = ':', label = 'subscriptions')
plt.plot(x,a4_ds,color='darkturquoise',linewidth=2, linestyle = ':', label = 'subscribers')
plt.plot(x,a5_ds,color='fuchsia',linewidth=2, linestyle = ':', label = 'videos')
plt.xlabel('degree')
plt.ylabel('network diameter')
plt.text(0,0 ,u'c')

plt.subplot(2,2,4)#, constrained_layout=True
plt.plot(x,a1_de,color='red',linewidth=2, label = 'target')
plt.plot(x,a2_de,color='tomato',linewidth=2, linestyle = ':', label='friends')
plt.plot(x,a3_de,color='yellow',linewidth=2, linestyle = ':', label = 'subscriptions')
plt.plot(x,a4_de,color='darkturquoise',linewidth=2, linestyle = ':', label = 'subscribers')
plt.plot(x,a5_de,color='fuchsia',linewidth=2, linestyle = ':', label = 'videos')
plt.xlabel('degree')
plt.ylabel('average degree')
plt.text(1,0, u'd')

plt.legend(loc=0, ncol=1, bbox_to_anchor=(2, 1))

plt.subplots_adjust(wspace =0.5, hspace =0.5)
plt.show()

fig.savefig('graph_statistics.svg', bbox_inches = 'tight')#, 