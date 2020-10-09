from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
from matplotlib import cm
from sklearn.cluster import MeanShift
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
io = r'data_file_path'
data = pd.read_excel(io, sheet_name = 0, header=0, index_col = 0, nrows = 69)

def k_silhouette(data,bandwidth):
  K1 = np.linspace (start,stop)
  S = []
  P= []
  for k in K1:
     meanshift=MeanShift(bandwidth=k, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True).fit(data)
     labels = meanshift.labels_
     b=max(labels)
     c=b+1
     P.append(int(c))
     S.append(metrics.silhouette_score(data, labels, metric='euclidean'))
  print("Silhouette Coefficient:",S)
  print("Number of clusters:",P)
  print("bandwidth:",K1)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(S, P, K1)
  ax.set_xlabel('Silhouette Coefficient')
  ax.set_ylabel('Number of clusters')
  ax.set_zlabel('bandwidth')
  zminorLocator=MultipleLocator(1)
  plt.show()
k_silhouette(data, bandwidth=stop)

def k_davies(data,bandwidth):
  K2 = np.linspace (start,stop)
  D = []
  M= []
  for k in K2:
     meanshift=MeanShift(bandwidth=k, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True).fit(data)
     labels = meanshift.labels_
     b=max(labels)
     c=b+1
     M.append(int(c))
     D.append(metrics.davies_bouldin_score(data, labels))
  print("Davies Bouldin:",D)
  print("Number of clusters:",M)
  print("bandwidth:",K2)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(D, M, K2)
  ax.set_xlabel('Davies Bouldin')
  ax.set_ylabel('Number of clusters')
  ax.set_zlabel('bandwidth')
  zminorLocator=MultipleLocator(1)
  plt.show()
k_davies(data, bandwidth=stop)

def k_calinski(data,bandwidth):
  K3 = np.linspace (start,stop)
  C = []
  N= []
  for k in K3:
     meanshift=MeanShift(bandwidth=k, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True).fit(data)
     labels = meanshift.labels_
     b=max(labels)
     c=b+1
     N.append(int(c))
     C.append(metrics.calinski_harabasz_score(data, labels))
  print("Calinski Harabasz:",C)
  print("Number of clusters:",N)
  print("bandwidth:",K3)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(C, N, K3)
  ax.set_xlabel('Calinski Harabasz')
  ax.set_ylabel('Number of clusters')
  ax.set_zlabel('bandwidth')
  zminorLocator=MultipleLocator(1)
  plt.show()
k_calinski(data, bandwidth=stop)
