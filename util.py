import numpy as np
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

##################################
##  Fission-Fusion framework    ##
##################################

def fission(X, betas, labels, OFM_index):
    """
    2-means to the fitted component with one-fit-many association
    """
    fitted_component = X[labels == OFM_index]
    centroids = np.delete(betas, OFM_index, axis=0) # delete the current centroid
    fission_centriods = KMeans(n_clusters=2,init="random",random_state = 0, algorithm='full',n_init=1).fit(fitted_component).cluster_centers_
    return np.vstack((centroids,fission_centriods))


def fussion(centroids, MFO_index):
    """
    Merge the two centers with many-fit-one association
    """
    fussion_centroid = (centroids[MFO_index[0]] + centroids[MFO_index[1]])/2
    centroids = np.delete(centroids, MFO_index, axis=0) 
    return np.vstack((centroids, fussion_centroid)) 

def FissionFusion(X, K, seed, ofm_method='mean_std', 
                  mfo_method='distance', init_method='random', 
                  radius_method='min', maxIter=30, start_epsilon=0.1,
                  ofm_adapt_shape=False, mfo_adapt_shape=False):
    """
    Proposed Fission Fusion kmeans procedure
    """
    store_C = []
    sse = []
    OFM_history = []
    MFO_history = []
    ### initial kmeans ####
    # make sure no almost empty component 
    while True:
        kms = KMeans(n_clusters=K, init=init_method,n_init=1,random_state=seed,algorithm='full').fit(X)
        labels = kms.labels_
        min_cluster_num = min([np.sum(labels==j) for j in range(K)])
        seed+=1
        if min_cluster_num >=10:
            break
        
    betas = kms.cluster_centers_  
    labels = kms.labels_
    sse.append(kms.inertia_) 
    store_C.append(betas)
    iterCount = 0
    while iterCount < maxIter:
        iterCount+=1
        #####step1: detect OFM #####
        if ofm_method == 'mean_std':
            OFM_index = detect_ofm_std(X, betas, labels, adapt_shape=ofm_adapt_shape)
        elif ofm_method == 'sum_std':
            OFM_index = detect_ofm_sum_std(X, betas, labels)
        elif ofm_method == 'radius':
            OFM_index = detect_ofm_radius(X, betas, labels, radius_method, start_epsilon=start_epsilon)     
        OFM_history.append(OFM_index)

        #####step2: Fission #####   
        betas = fission(X, betas, labels, OFM_index)
        
        labels = compute_Voronoi(X, betas)
        #####step3: detect MFO #####
        
        if mfo_method == 'distance':
            MFO_index = detect_mfo_distance(X, labels, betas,adapt_shape=mfo_adapt_shape)
        if mfo_method == 'voronoi':
            MFO_index = detect_mfo_Voronoi(X, betas)
        if mfo_method == 'cost':
            SSEDM_list = cal_SSEDM(X, betas, labels)
            SSEDM2_list, second_index = cal_dis_seC(X, betas, labels)
            Cost = abs(np.array(SSEDM_list) - np.array(SSEDM2_list)).tolist()
            MFO_index = [Cost.index(min(Cost)), np.argsort(Cost)[1]] #[minima index second minima index]   

        MFO_history.append(MFO_index)
        

        #####step4 : Fusion #####
        betas = fussion(betas, MFO_index)
        
        assert len(betas) == K # invariance
        
        #####step5: k-means adjustment #####
        kms = KMeans(n_clusters=K, init=betas, random_state=seed, n_init=1, max_iter=3).fit(X)
        betas = kms.cluster_centers_
        labels = kms.labels_
        new_sse = kms.inertia_
        if sse[-1] <= new_sse: 
            break
        else:
            sse.append(new_sse)
            store_C.append(betas) 

    return store_C,sse,seed, OFM_history, MFO_history
    
    
##################################
##  Detection ofm subroutine    ##
################################## 

# Method1: STD 

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
        std_list.append(np.sqrt(mean_squared_distance(data=ptsInCluster,center=cluster_center))/max_dist_to_center)    
        #std_list.append(sum_squared_distance(data=ptsInCluster,center=cluster_center))
    OFM_index = std_list.index(max(std_list))  
    return OFM_index




# Method1: Radius##

def detect_ofm_radius(X, betas, labels, radius_method, start_epsilon):
    """ output the cluster index with the smallest point containment ratio
    within a ball centered at the cluster center, with radius epsilon 
    epsilon is automatically adjusted here. 
    """
    #calculate base radius
    m_dist = []
    for center_index, cluster_center in enumerate(betas):
        ptsInCluster = X[labels == center_index]
        m_dist.append(np.sqrt(median_squared_distance(data=ptsInCluster,center=cluster_center)))
    epsilon = start_epsilon
    while True:
        if radius_method == 'min':
            radius =  np.min(m_dist) * epsilon
        if radius_method == 'median':
            radius =  np.median(m_dist) * epsilon
        num_point_within_radius = []
        cluster_size = []
        for center_index, cluster_center in enumerate(betas):
            ptsInCluster = X[labels == center_index]
            num_point_within_radius.append(num_points_by_radius(data=ptsInCluster, center=cluster_center, radius=radius))
            cluster_size.append(len(ptsInCluster))
        if max(num_point_within_radius) <=10: # if the number is small, then it might induce noise in estimation
            epsilon = epsilon * 2
        else:
            break
    num_point_within_radius_percentage = [num_point_within_radius[i]/cluster_size[i] for i in range(len(betas))]
    OFM_index = num_point_within_radius_percentage.index(min(num_point_within_radius_percentage)) 
    #print ("num_points_within", num_point_within_radius, "cluster size", cluster_size)
    return OFM_index

    
#############################################
##  Detection mfo subroutine: Pairwise    ###
#############################################

# Method1: Distance

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



# Method1: Voronoi

def detect_mfo_Voronoi(X, labels, betas):
    """ output the indices of two clusters whose Voronoi 
    perturbation ratio is the largest.    
    The perturbed distance is automatically adjusted
    """
    max_perturbed_ratio = 0
    k = len(betas)
    plt.figure(figsize=(16,10))
    plt.subplot(2,2,1)
    plt.scatter(X[:,0], X[:,1], c='0.6')
    plt.scatter(betas[:,0], betas[:,1], c='r')
    for l in range(k):
        plt.text(betas[l,0]+1, betas[l,1]+1, str(l))
    epsilon = 0.001
    while max_perturbed_ratio == 0:
        for i in range(k-1):
            for j in range(i+1,k):
                sub_X = X[(labels==i) | (labels==j)]
                sub_labels = compute_Voronoi(sub_X, betas[[i,j],:])
                voronoi_i = list(np.argwhere(sub_labels==0).reshape(-1))
                voronoi_j = list(np.argwhere(sub_labels==1).reshape(-1))
                direction = betas[j]-betas[i]
                sub_perturbed_betas = betas[[i,j],:] + epsilon * direction
                new_sub_labels = compute_Voronoi(sub_X, sub_perturbed_betas)
                new_voronoi_i = list(np.argwhere(new_sub_labels==0).reshape(-1))

                change_size = len(set(new_voronoi_i).symmetric_difference(set(voronoi_i)))

                perturbed_ratio = min(change_size/len(voronoi_i),change_size/len(voronoi_j))
                if perturbed_ratio > max_perturbed_ratio:
                    max_perturbed_ratio = perturbed_ratio
                    MFO_index = [i,j]
        if max_perturbed_ratio == 0:
            epsilon = 2 * epsilon
    return MFO_index


#############################
##  Suppliment Function   ###
#############################

def compute_Voronoi(X, betas):
    """
    Return the index of the closest centroid for each data point
    """
    for index, beta in enumerate(betas):
        if index == 0:
            dist_vec = np.sum((X - beta)**2, axis=1).reshape((-1,1))
        else:
            dist_vec = np.concatenate((dist_vec, np.sum((X - beta)**2, axis=1).reshape((-1,1))), axis=1)
    assert dist_vec.shape == (len(X), len(betas))
    labels = np.argmin(dist_vec, axis=1)
    return labels

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

def median_squared_distance(data, center):
    """ Input
        - data : np.array, of shape N*d
        - center: 1*d
        Output
        - median squared distance of data to the center
    """
    return np.median(np.sum(np.power(data - center, 2), axis=1))

def num_points_by_radius(data, center, radius):
    """ Input
        - data : np.array, of shape N*d
        - center: 1*d
        Output
        - number of data points whose distance to the center is smaller than radius
    """
    return np.sum(np.sum(np.power(data - center, 2), axis=1) <= radius**2)
    


    
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


# Method 2: result centroids != ground truth centroids

def Centroid_Index_diff(gt,centroids):
    """
    compute centroid index with different cluster number
    only consider candidate centroids mappting to ground truth. 
    each missed ground truth will make CI +=1
    """
    CI_list = []
    CI = 0
    #-----------------------------------red to blue(centroid mapping to groud truth)-----------------------------------
    for index, gt_center in enumerate(gt):
        distance_to_centroids = np.sum((centroids - gt_center)**2, axis=1)
        min_dist_to_centroid_index = np.argmin(distance_to_centroids)
        if (min_dist_to_centroid_index in CI_list):
            CI+=1
        else:
            CI_list.append(min_dist_to_centroid_index)
    return CI

############################
###      k-means -+      ###
############################
def cal_SSEDM(X, betas, labels):
    """ output the every cluster SSEDM """
    SSEDM_list = []
    
    for center_index, cluster_center in enumerate(betas):
        ptsInCluster = X[labels == center_index]
        SSEDM_list.append(sum_squared_distance(data=ptsInCluster,center=cluster_center))
    return SSEDM_list

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

'''
re-writed in k-means-+
'''
def sum_squared_distance(data, center):
    """ Input
        - data : np.array, of shape N*d
        - center: 1*d
        Output
        - sum squared distance of data to the center
    """
    return np.sum(np.sum(np.power(data - center, 2), axis=1))

def detect_ofm_sum_std(X, betas, labels):
    """ output the cluster index with the largest mean squared distance """
    std_list = []
    for center_index, cluster_center in enumerate(betas):
        ptsInCluster = X[labels == center_index]
        std_list.append(sum_squared_distance(data=ptsInCluster,center=cluster_center))
    OFM_index = std_list.index(max(std_list))  
    return OFM_index


def Swap(X, K, seed, maxIter=30):                 
    """
    Swap Procedure:
        [1]Detect one centroid to move           [in our setting ofm]
        [2]Detect another centorid to do 2-kmeans[in our setting mfo]
        [3]Swap                                  [Delete, add]
        [4]re-clustering again(limited k-means iter < 50)
        Termination condition                    [SSE increase then stop]
    """ 
    #List store
    store_C = []
    sse = []
    swap_history = []
    
    #step1. k-means, paper applied a enhanced.
    kms = KMeans(n_clusters=K, init='random',n_init=1,random_state=seed,algorithm='full').fit(X)
    
    betas = kms.cluster_centers_  
    labels = kms.labels_
    sse.append(kms.inertia_) 
    store_C.append(betas)
    iterCount = 0
    
    while iterCount < maxIter:
        iterCount += 1
        #step1. find ofm index(std)
        swap1_index = detect_ofm_std(X, betas, labels)
        #step2. find mfo index(distance)
        '''
        MFO_index = detect_mfo_distance(X, labels, betas)
        swap2_index = random.choice(MFO_index)
        '''
        #Cost
        SSEDM_list = cal_SSEDM(X, betas, labels)
        SSEDM2_list, second_index = cal_dis_seC(X, betas, labels)
        Cost = abs(np.array(SSEDM_list) - np.array(SSEDM2_list)).tolist()
        swap2_index = Cost.index(min(Cost))
        
        #step3. swap
        v = 0.01
        swap1 = betas[swap1_index] + v                #ofm centroid nearby
        betas = np.delete(betas, swap2_index, axis=0) #delete mfo
        betas = np.vstack((betas,swap1))             #comconcatenate swap1
        #oberseve
        swap_history.append([swap1_index, swap2_index])
        
        #####step4: k-means adjustment #####
        kms1 = KMeans(n_clusters=K, init=betas, n_init=1, max_iter=3).fit(X)
        betas = kms1.cluster_centers_
        labels = kms1.labels_
        new_sse = kms1.inertia_

        if sse[-1] < new_sse: 
            break
        else:
            sse.append(new_sse)
            store_C.append(betas)         
    return store_C, sse, seed, swap_history, new_sse
############################
############################
