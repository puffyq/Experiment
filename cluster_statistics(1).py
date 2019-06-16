# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:37:54 2019

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax

fig = plt.figure(num=1, figsize=(5, 5))
colors = ['red','tomato','yellow','green']
label = ['neighbors<=0', '0<neighbors<=1', '1<neighbors<=5', 'neighbors>5']
per = [0.363,	0.174,	0.259, 0.203]
explode = [0.1, 0, 0, 0] # 0.1 凸出这部分，
plt.axes(aspect=1)  # set this , Figure is round, otherwise it is an ellipse
#autopct ，show percet
plt.pie(x=per, explode=explode,labels = label, colors = colors, autopct='%3.1f %%',
        shadow=True, labeldistance=1.1, startangle = 90,pctdistance = 0.6 )

plt.legend(loc='upper left', bbox_to_anchor=(-0.4, 1))
fig.savefig('pie_chart.svg',bbox_inches = 'tight')

plt.show()

fig = plt.figure(num=1, figsize=(8, 5))
#colors = ['red','tomato','yellow']
#label = ['most friends', '2nd most friends', '3nd most friends']
a1nd = [2.98, 6.88, 7.88, 4.79]
a2nd = [2.68, 1.016, 8.44, 2.4]
a3nd = [2.33, 1, 4.09, 2]

index = np.array([0, 1, 2, 3])

bar_width = 0.2
plt.bar(index , a1nd, width=0.2 , color='red' , label = 'most neighbors')
plt.bar(index +  bar_width, a2nd, width=0.2 , color='tomato', label = '2nd most neighbors')
plt.bar(index +  2*bar_width, a3nd, width=0.2 , color='yellow', label = '3nd most neighbors')

plt.xticks( index+0.22, ('a2','a3','a4','a5') )

plt.xlabel('subnet')
plt.ylabel('Edge weights')

plt.legend(loc=0, ncol=1)
fig.savefig('top3.svg',bbox_inches = 'tight')
plt.show()

fig = plt.figure(num=1, figsize=(8, 5))
c0 = [2.52, 2.54, 2.59, 2.9]
c1 = [1.26, 1.3, 1.43, 2.02]
c2 = [1.15, 1.26, 1.35, 2.36]
c3 = [3.26, 3.43, 3.75, 3.87]

index = np.array([0, 1, 2, 3])

bar_width = 0.2
plt.bar(index , c0, width=0.2 , color='red' , label = u'mutual friends net')
plt.bar(index +  bar_width, c1, width=0.2 , color='tomato', label = u'shared subscriptions net')
plt.bar(index +  2*bar_width, c2, width=0.2 , color='yellow', label = u'shared subscripers net')
plt.bar(index +  3*bar_width, c3, width=0.2 , color='green', label = u'shared videos net')

plt.xticks( index+0.3,('0','1','2','3') )

plt.xlabel('cluster')
plt.ylabel('Edge weights')

plt.legend(loc=0, ncol=1)
fig.savefig('cluster.svg',bbox_inches = 'tight')
plt.show()
