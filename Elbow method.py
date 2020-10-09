import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
io = r'data_file_path'
data = pd.read_excel(io, sheet_name = 0, header=0, index_col = 0, nrows = 69)

sse_list = [] 
K = range(1,17) 
for k in range(1,17): 
    kmeans_model = KMeans(n_clusters=k, random_state=0).fit(data) 
    sse_list.append(kmeans_model.inertia_)   
print(sse_list)

plt.figure()
plt.style.use('ggplot') 
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rc('font',family='Times New Roman')
plt.rcParams['axes.unicode_minus'] = False      
plt.plot(K, sse_list, 'b*-')
plt.ylabel('SSE')
plt.xlabel('Number of clusters')
plt.style.use('ggplot') 
plt.show()
