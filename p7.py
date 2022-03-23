

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm

from sklearn import preprocessing #includes scaling, centering, normalization, 
                                    #binarization and imputation methods.
from sklearn.mixture import GaussianMixture
#a probabilistic model for representing normally distributed subpopulations
#within an overall population

import pandas as pd
import numpy as np

iris = datasets.load_iris()

X = pd.DataFrame(iris.data)
X.columns = ['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
#print("X::\n",X)

y = pd.DataFrame(iris.target)
y.columns = ['Targets']
#print("y::\n",y)


#k-means algorithm
model = KMeans(n_clusters=3)
model.fit(X) # computes k means clustering
#print(model.labels_)
score1=sm.accuracy_score(y, model.labels_)
print("Accuracy of KMeans=",score1)

plt.figure(figsize=(7,7))
colormap = np.array(['red', 'lime', 'black'])
plt.subplot(1, 2, 1) # row,col,index
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')

#EM algorithm
#Standardize features by removing the mean and scaling to unit variance
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
#print(xsa)
xs = pd.DataFrame(xsa, columns = X.columns)
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)                   #Estimate model parameters with the EM algorithm.
y_cluster_gmm = gmm.predict(xs)     #Predict the labels for the data samples 
                                    #in X using trained model.
#print(y_cluster_gmm)
score2=sm.accuracy_score(y, y_cluster_gmm)
print("Accuracy of EM=",score2)
plt.subplot(1, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[y_cluster_gmm], s=40)
plt.title('EM Classification')
