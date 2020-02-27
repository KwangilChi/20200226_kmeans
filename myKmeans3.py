# -*- coding: utf-8 -*-

#### Generate random data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=20, centers=3, n_features=2,
                  random_state=3)

import numpy as np
import matplotlib.pyplot as plt


#### Inertia is the mean squared distance between each instance and its closest centroid.
def inertia(X, centroids, labels):
    sqd_list=[]
    for i in range(3):
        sqd_list.append((([X[labels==j] for j in range(3)][i]-centroids[i])**2).mean())
    inertia = (np.mean(sqd_list))
    return inertia

#### My k_means function
def k_means(X, k, n_iter):
                   
#### Enter a for loop
    for i in range(1,n_iter):
#### Initialise centroids randomly using numpy randint to select k number of integers from 0 to n.    
        centroids = X[np.random.randint(low=0, high=len(X), size=k)]
        inertia_list = []
        centroid_list = []
        label_list = []
        
        while True:
            centroid_list.append(centroids)
            
#### Calculate the euclidean distance between each point and each of the centroids, and assign label. 
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2)) 
            labels = np.argmin(distances, axis=0)
            
            label_list.append(labels)
            
#### Find the mean of each cluster. These are my new centroids
            for i in range(0,k):
                if X[labels==i].size !=0:
                        new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(k)]) 
            inertia_list.append(inertia(X, centroids, labels))
#### Condition to break the while loop is when my new_centroids are equal to the set centroids
            if np.all(centroids == new_centroids):
                break
            
            centroids = new_centroids
#### Find the index of the minimum inertia 
        idx = np.argmin(inertia_list)
        centroids = centroid_list[idx]
        labels = label_list[idx]
        
    return centroids, labels

#### Function to plot my clusters and centroids
def plot_kmeans(X, centroids, labels):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow') 
    plt.scatter(centroids[:,0] ,centroids[:,1], color='black')
    
    
##### Function to plot my clusters and centroids
#def elbow_method(X, Kmax):
#    inertias = [] 
#    K = range(2,Kmax) 
##### Calculate the inertias  
#    for k in K: 
#        centroids, labels = k_means(X,k)
#        inertias.append(inertia(X,centroids)) 
#    

a, b = k_means(X, 3, 10); plot_kmeans(X, a, b)
