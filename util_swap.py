import numpy as np
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt


def cal_SSEDM(X, betas, labels):
    """ output the every cluster SSEDM """
    SSEDM_list = []
    
    for center_index, cluster_center in enumerate(betas):
        ptsInCluster = X[labels == center_index]
        SSEDM_list.append(sum_squared_distance(data=ptsInCluster,center=cluster_center))
    return SSEDM_list


def sum_squared_distance(data, center):
    """ Input
        - data : np.array, of shape N*d
        - center: 1*d
        Output
        - sum squared distance of data to the center
    """
    return np.sum(np.sum(np.power(data - center, 2), axis=1))

def fission(X, betas, labels, OFM_index):
    """
    2-means
    """
    if len(fitted_component) > 1: 
        centroids = np.delete(betas, OFM_index, axis=0) # delete the current centroid
        fission_centriods = KMeans(n_clusters=2,init="random",random_state = 0, algorithm='full',n_init=1).fit(fitted_component).cluster_centers_
        return np.vstack((centroids,fission_centriods))

def cal_dis_seC(X, betas, labels):
    second_index=[]
    SSEDM2_list = []
    
    for center_index in range(len(betas)):
        ptsInCluster = X[labels == center_index]
        dist = []
        for i in range(len(betas)):
            dist.append(sum_squared_distance(ptsInCluster,betas[i]))
        second_center = np.argsort(dist)[1] #find second minima index of list
        second_index.append(second_center)
        SSEDM2_list.append(sum_squared_distance(data=ptsInCluster,center=betas[second_center]))
    
    return SSEDM2_list, second_index


def detect_ofm_std(X, betas, labels, adapt_shape=False):
    """ output the cluster index with the largest mean squared distance """
    std_list = []
    for center_index, cluster_center in enumerate(betas):
        ptsInCluster = X[labels == center_index]
        if adapt_shape:
            # normalize each cluster with the same maximal radius
            max_dist_to_center = np.sqrt(max_squared_distance(data=ptsInCluster,center=cluster_center))
        else:
            max_dist_to_center = 1
        #std_list.append(np.sqrt(mean_squared_distance(data=ptsInCluster,center=cluster_center))/max_dist_to_center)    
        std_list.append(sum_squared_distance(data=ptsInCluster,center=cluster_center))
    OFM_index = std_list.index(max(std_list))  
    return OFM_index


def mean_squared_distance(data, center):
    """ Input
        - data : np.array, of shape N*d
        - center: 1*d
        Output
        - mean squared distance of data to the center
    """
    return np.mean(np.sum(np.power(data - center, 2), axis=1))
    
def max_squared_distance(data, center):
    """ Input
        - data : np.array, of shape N*d
        - center: 1*d
        Output
        - max squared distance of data to the center
    """
    return np.max(np.sum(np.power(data - center, 2), axis=1))

def detect_mfo_distance(X, labels, betas, adapt_shape=False):
    """ output the indices of a pair of clusters whose distance is the smallest"""
    k = len(betas)
    min_dist = np.inf
    for i in range(k-1):
        for j in range(i+1,k):
            distance = np.sum((betas[i,:] - betas[j,:])**2)
            if adapt_shape:
                radius_cluster_i = max_squared_distance(X[labels==i], betas[i,:])
                radius_cluster_j = max_squared_distance(X[labels==j], betas[j,:])
                max_radius = max_squared_distance(X[labels==i | labels==j], (betas[i,:]+betas[j,:])/2)
                distance = distance / max_radius # normalize the pairwise distance by the radius
            if distance < min_dist:
                MFO_index = [i,j]
                min_dist = distance
    return MFO_index

###########################
## Centroid Index Metric ##
###########################

# Method 1: result centroids = ground truth centroids

def Centroid_Index(gt,centroids):
    """
    compute centroid index
    """
    if len(gt) !=len(centroids):
        return 100
    else:
        k = len(gt)
        CI1_list = [] 
        CI1 = 0
        CI2_list = [] 
        CI2 = 0
        #-----------------------------------red to blue(groud truth to centroid)-----------------------------------
        for index, centroid in enumerate(centroids):
            distance_to_gt = np.sum((gt - centroid)**2, axis=1)
            min_dist_to_gt_index = np.argmin(distance_to_gt)
            if (min_dist_to_gt_index in CI1_list): 
                CI1+=1
            else:
                CI1_list.append(min_dist_to_gt_index)
        #-----------------------------------blue to red(centroid to groud truth)-------------------------       
        for index, gt_center in enumerate(gt):
            distance_to_centroids = np.sum((centroids - gt_center)**2, axis=1)
            min_dist_to_centroid_index = np.argmin(distance_to_centroids)
            if (min_dist_to_centroid_index in CI2_list):
                CI2+=1
            else:
                CI2_list.append(min_dist_to_centroid_index)
          
        return max(CI1,CI2)




 