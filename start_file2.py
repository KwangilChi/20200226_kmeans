# -*- coding: utf-8 -*-

#### Generate random data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=20, centers=5, n_features=2,
                  random_state=3)

import numpy as np
import matplotlib.pyplot as plt

#### My k_means function
def k_means(X, k):

#### Initialise centroids randomly using numpy randint to select k number of integers from 0 to n.    
    centroids = X[np.random.randint(low=0,
                                    high=len(X), 
                                    size=k)]                     

#### Enter while loop.
    while True:
        
#### Calculate the euclidean distance between each point and each of the centroids, and assign label. 
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2)) 
        labels = np.argmin(distances, axis=0)
        
#### Find the mean of each cluster. These are my new centroids
        new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(k)]) 

#### Condition to break the while loop is when my new_centroids are equal to the set centroids
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
        
    return centroids, labels

#### Inertia is the mean squared distance between each instance and its closest centroid.
def inertia(X, centroids):
    sqd_list=[]
    for i in range(3):
        sqd_list.append((([X[labels==j] for j in range(3)][i]-centroids[i])**2).mean())
    inertia = (np.mean(sqd_list))
    return inertia

#### Function to plot my clusters and centroids
def plot_kmeans(X, centroids):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow') 
    plt.scatter(centroids[:,0] ,centroids[:,1], color='black')
    
    
#### Function to plot my clusters and centroids
def elbow_method(X, Kmax):
    inertias = [] 
    K = range(2,Kmax) 
#### Calculate the inertias  
    for k in K: 
        centroids, labels = k_means(X,k)
        inertias.append(inertia(X,centroids)) 
    
    
    
inertias = [] 

K = range(1,10) 
  
for k in K: 
    centroids, labels = k_means(X,k)
    inertias.append(inertia(X,centroids)) 
