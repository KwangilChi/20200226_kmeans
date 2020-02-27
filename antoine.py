# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


#### make data
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=1000, centers=5, n_features=2,random_state=1)


def random_c(C, X, number_clusters):
    """ Return number_clusters random centroids from dataset X in order 
    to inialize K-means classification. Retruns nothing, array C is directly modified."""
    C[0]=X[np.random.randint(low=0, high=len(X), size=number_clusters)]

def euclidian(A,B):
    """ Return the euclidian distance between A and B."""
    return np.linalg.norm(A-B)

def min_euclidian(C,X,labels,number_clusters):
    """ Given centroids in C[0], dataset X, cluster matrix K and number of clusters,
    returns the data points in X in the appropriate cluster K[x]."""
    i=0
    #We 'reset' K since data points may be assigned to another cluster.
    for xi in X:
        #distance is set anormally high to be tested.
        distance=10**6
        cluster=-1
        j=0
        for cj in C[0]:
            if euclidian(xi,cj)<distance:
                distance=euclidian(xi,cj)
                cluster=j
            j+=1
        labels[i]=cluster
        i+=1
    return labels

def mean_points_in_clusters(C,X,labels,number_clusters):
    """Define new centroids in C[0] being the barycenter of all data points
    in each cluster."""
    for cluster in range(0,number_clusters):
        if X[:,0][labels==cluster].size != 0:
            C[0][cluster][0]=X[:,0][labels==cluster].mean()
            C[0][cluster][1]=X[:,1][labels==cluster].mean()


def inertia_calculation(C,X,labels,number_clusters):
    """ Returns inertia of classification. 
    Inertia is the mean squared distance between each instance and 
    its closest centroid.""" 
    inertia=0
    for cluster in range(0,number_clusters):
        inertia+=euclidian(X[labels==cluster],C[0][cluster])
    return inertia/len(X)
        
def plot_clusters(C,X,labels):
    """ Plot clusters a,d their centroids. Clusters will be sorted by labels."""
    colors=('g','b','pink','y','m','cyan')
    # plt.scatter(X[:,0],X[:,1],alpha=0.4)
    for i in range(0,C.shape[1]):
        plt.scatter(X[:,0][labels==i],X[:,1][labels==i],c=colors[i],alpha=0.3)
    for i in range(0,len(C[0])):
        plt.scatter(C[0][i][0],C[0][i][1],c='r')
#### expected function

def k_means_algo(X, number_clusters,n_init):
    """Probably not optimal K-means algorithm. The best inertia will be selected out of
    n_init iterations."""
    #inertia is set anormally high to be tested for lower value.
    inertia=float(10**6)
    for i in range(0,n_init):
        labels=np.zeros(len(X))
        SameCentroids=False
        #Creating a numpy array for sotring centroids. Two lines for after/before comparison
        C=np.zeros((2,number_clusters,2)) 
        #Storing random centroids in first line of array C
        random_c(C,X, number_clusters) 
        #Creating clusters, filling with points closest to centroids. Shape is number of clusters * length X. If filled with 0, no points.
        labels=min_euclidian(C,X,labels,number_clusters)
        
        while SameCentroids==False:
            #Second line of C becomes the centroids to compare to see if the algorithm had converged
            C[1]=C[0]
            #Defining new centroids as average of points in each clusters :
            mean_points_in_clusters(C,X,labels,number_clusters)
            labels=min_euclidian(C,X,labels,number_clusters)
            if (C[0]==C[1]).all():
                SameCentroids=True
        #Si l'iniertie de cette boucle est inférieure aux précédentes, on stocke les résultats optimaux.
        if inertia_calculation(C,X,labels,number_clusters)<inertia:
            inertia=inertia_calculation(C,X,labels,number_clusters)
            Copt=C
            labels_opt=labels
    return Copt,labels_opt

number_clusters=3
C,labels=k_means_algo(X,number_clusters,15)
plot_clusters(C,X,labels)