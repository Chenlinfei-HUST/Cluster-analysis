import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
io = r'data_file_path'
data = pd.read_excel(io, sheet_name = 0, header=0, index_col = 0, nrows = 69)

X = pd.DataFrame(data)

res = []
for eps in np.arange(0.01,2.1,0.01):  
    for min_samples in range(1,3):
        dbscan = DBSCAN(eps = eps, min_samples = min_samples,metric='euclidean', algorithm='auto').fit(data)
        labels = dbscan.labels_
        n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
        outliners = np.sum(np.where(dbscan.labels_ == -1,0,1)==0) 
        stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
        CH=metrics.calinski_harabasz_score(data, labels)
        DB=metrics.davies_bouldin_score(data, labels)
        SC=metrics.silhouette_score(data, labels, metric='euclidean')
        res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners,'CH':CH,'DB':DB,'SC':SC})
      
df = pd.DataFrame(res)
df_select2=df.loc[df.n_clusters == 2, :]
df_select3=df.loc[df.n_clusters == 3, :]
df_select4=df.loc[df.n_clusters == 4, :]
df_select5=df.loc[df.n_clusters == 5, :]
df_select6=df.loc[df.n_clusters == 6, :]
df_select7=df.loc[df.n_clusters == 7, :]
df_select8=df.loc[df.n_clusters == 8, :]
print("2 clusters:\n",df_select2)
print("3 clusters:\n",df_select3)
print("4 clusters:\n",df_select4)
print("5 clusters:\n",df_select5)
print("6 clusters:\n",df_select6)
print("7 clusters:\n",df_select7)
print("8 clusters:\n",df_select8)
