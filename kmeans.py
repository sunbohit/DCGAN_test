from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # mpimg 用于读取图片

import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import os

import time

start = time.time()

np.random.seed(42)

path = "./data/lovelive_192_192"
files= os.listdir(path)
count = 0
data = []
filename = []
for file in files: 
    count += 1
    picture = mpimg.imread(os.path.join(path,file)) 
    #print(picture.shape)
    #print(file)
    feature = np.reshape(picture,[192*192*3,])
    #print(feature.shape)
    data.append(feature)
    filename.append(file)
    if (count > 99):
        break

data = scale(data)
print(data.shape)

k = 9

epochs = 100

kmeans_model = KMeans(n_clusters=k, max_iter=epochs, n_jobs=8)
kmeans_model.fit(data)

print(kmeans_model.labels_)
s1 = pd.Series(kmeans_model.labels_)
s1 = s1.value_counts()
print(s1)

end = time.time()

hours = ((end-start)/60/60)
print(hours)