# -*- coding: utf-8 -*-

#### make data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=20, centers=3, n_features=2,
                  random_state=11)

import numpy as np
import matplotlib.pyplot as plt

#### expected function
def k_means_algo(X, k, randstate):
    rs = np.random.RandomState(randstate) # Random state generator
    i = rs.permutation(len(X))[:k]        # Returns random k number of indices
    centroids = X[i]                      # Using the indices initialised above, define centroids

    while True:
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2)) # Calculate the euclidean distance between each point and each of the centroids, and assign label. 
        labels = np.argmin(distances, axis=0)
        
        new_centroids = np.array([X[labels==i].mean(axis=0) for i in range(k)]) # Finds new centroids depending on their distance with centroid.
        
        if np.all(centroids == centroids):
            break
        
        centroids = new_centroids
        
    return centroids, labels

def plot_kmeans(X, centroids):
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow') 
    plt.scatter(centroids[:,0] ,centroids[:,1], color='black')