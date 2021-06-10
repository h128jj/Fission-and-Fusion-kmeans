import math
import random
import time
import numpy as np
import pandas as pd
import xlwt
from scipy.spatial.distance import cdist
from scipy.cluster.vq import vq
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans  
from sklearn.datasets import make_blobs
from matplotlib.pyplot import *
from pylab import *
import matplotlib.pyplot as plt
####
from util import *
import warnings
warnings.filterwarnings('ignore')


class Fmeans:
    def __init__(self, ofm_method='std', 
                       mfo_method='distance', 
                       init_method='random', 
                       radius_method='min', 
                       maxIter=50, 
                       start_epsilon=0.1,
                       ofm_adapt_shape=False, 
                       mfo_adapt_shape=False):
        
        self.init_method = init_method
        self.ofm_method = ofm_method
        self.mfo_method = mfo_method
        
    def fit(self, X, k, seed):
        store_C, sse, seed, OFM_history, MFO_history = FissionFusion(X, k, seed, 
                                                           ofm_method=self.ofm_method, 
                                                           mfo_method=self.mfo_method,
                                                           init_method=self.init_method, 
                                                           maxIter=2*k)
    
        
        self.labels_ = compute_Voronoi(X, store_C[-1])
        self.cluster_centers_ = store_C[-1]
        self.inertia_ = sse[-1]
        
    def fit_related(self, X, k, seed, method = 'X-means', start_k=2, kmax=5):
        '''
        method: X-meanx, Alg7, Alg8
        start_k: Applied when method = X-means, control init centroids number
        kmax:    Applied when method = Alg7,    control max split number
        '''
        if method == 'X-means':
            '''
            (2000) X-means: Extending K-means with Efficient Estimation of the Number of Clusters.
            code source: https://github.com/alex000kim/XMeans
            '''
            xm = XMeans()
            xm.fit(X,start_k)
            self.labels_ = compute_Voronoi(X, xm.cluster_centers_)
            self.cluster_centers_ = [xm.cluster_centers_]
            self.inertia_ = [xm.inertia_]
        if method == 'Alg7':
            '''
            (2006) Clustering by the K-Means Algorithm Using a Split and Merge Procedure
            '''
            store_C, sse = Frce_kmeans(X, k, seed=1, kmax=5)
            self.labels_ = compute_Voronoi(X, store_C[-1])
            self.cluster_centers_ = store_C[-1]
            self.inertia_ = sse[-1]
        if method == 'Alg8':
            '''
            (2016) Robust K-means algorithm with automatically splitting and merging clusters and its applications for surveillance data
            '''
            store_C, sse, k_observe = Ds_kmeans(X, k, seed=1, batch_merge=False)
            self.labels_ = compute_Voronoi(X, store_C[-1])
            self.cluster_centers_ = store_C[-1]
            self.inertia_ = sse[-1]        