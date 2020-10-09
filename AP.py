from mpl_toolkits.mplot3d import Axes3D  
from sklearn.cluster import AffinityPropagation
import pandas as pd
from matplotlib import cm
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
io = r'data_file_path'
data = pd.read_excel(io, sheet_name = 0, header=0, index_col = 0, nrows = 69)


def k_effect(data,preference):
 M = range(start,stop,step)
 X= np.arange (0.5, 1, 0.1)
 H_list = []
 P_list= []
 D_list = []
 S_list = []
 for x in X:
     h=[]
     p=[]
     d=[]
     s=[]
     for m in M:
          AP = AffinityPropagation(damping=x, max_iter=200, convergence_iter=15, copy=True, preference=m, affinity='euclidean', verbose=False).fit(data)
          labels = AP.labels_
          b=max(labels)
          c=b+1
          p.append(int(c))
          h.append(metrics.calinski_harabasz_score(data, labels))
          d.append(metrics.davies_bouldin_score(data, labels))
          s.append(metrics.silhouette_score(data, labels, metric='euclidean'))
     P_list.append(p)
     H_list.append(h)
     D_list.append(d)
     S_list.append(s)
 S=np.array(S_list)
 P=np.array(P_list)
 D=np.array(D_list)
 H=np.array(H_list)
 print("Silhouette Coefficient:",S)
 print("Davies Bouldin:",D)
 print("Calinski Harabasz:",H)
 print("Number of clusters:",P)
 plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
 plt.rcParams['axes.unicode_minus'] = False
 fig = plt.figure()
 ax1 = fig.add_subplot(221)
 ax1.set_xlabel('Number of clusters')
 ax1.set_ylabel('Calinski Harabasz')
 ax1.scatter(P, H, s=1, c='b', marker='.')
 ax2 = fig.add_subplot(222)
 ax2.set_xlabel('Number of clusters')
 ax2.set_ylabel('Davies Bouldin')
 ax2.scatter(P, D, s=1, c='b', marker='.')
 ax3 = fig.add_subplot(223)
 ax3.set_xlabel('Number of clusters')
 ax3.set_ylabel('Silhouette Coefficient')
 ax3.scatter(P, S, s=1, c='b', marker='.')
 plt.show()
k_effect(data, preference=stop)
