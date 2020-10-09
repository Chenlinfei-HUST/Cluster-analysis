import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
io = r'data_file_path'
data = pd.read_excel(io, sheet_name = 0, header=0, index_col = 0, nrows = 69)

def k_silhouette(data,n_components):
  K = range (2,n_components+1)
  S = []
  for k in K:
     gmm=GaussianMixture(n_components=k).fit(data)
     labels = gmm.predict(data)
     S.append(metrics.silhouette_score(data, labels, metric='euclidean'))
  plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
  plt.rcParams['axes.unicode_minus'] = False
  plt.style.use('ggplot') 
  plt.plot(K, S, 'b*-')
  plt.xlabel('Number of clusters')
  plt.ylabel('Silhouette Coefficient')
  plt.show()
  print(S)
k_silhouette(data, n_components=8)

def k_davies(data,n_clusters):
  K = range(2,n_clusters+1)
  D = []
  for k in K:
     gmm=GaussianMixture(n_components=k).fit(data)
     labels = gmm.predict(data)
     D.append(metrics.davies_bouldin_score(data, labels))
  plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
  plt.rcParams['axes.unicode_minus'] = False
  plt.style.use('ggplot') 
  plt.plot(K, D, 'b*-')
  plt.xlabel('Number of clusters')
  plt.ylabel('Davies Bouldin')
  plt.show()
  print(D)
k_davies(data, n_clusters=8)

def k_calinski(data,n_clusters):
  K = range(2,n_clusters+1)
  C = []
  for k in K:
     gmm=GaussianMixture(n_components=k).fit(data)
     labels = gmm.predict(data)
     C.append(metrics.calinski_harabasz_score(data, labels))
  plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
  plt.rcParams['axes.unicode_minus'] = False
  plt.style.use('ggplot') 
  plt.plot(K, C, 'b*-')
  plt.xlabel('Number of clusters')
  plt.ylabel('Calinski Harabasz')
  plt.show()
  print(C)
k_calinski(data, n_clusters=8)
