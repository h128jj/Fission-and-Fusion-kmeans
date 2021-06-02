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
    dim = X.shape[1]
    fitted_component = X[labels == OFM_index]
    #judge sample in cluster number
    if len(fitted_component) > 1: 
        centroids = np.delete(betas, OFM_index, axis=0) # delete the current centroid
        fission_centriods = KMeans(n_clusters=2,init="random",random_state = 0, algorithm='full',n_init=1).fit(fitted_component).cluster_centers_
        return np.vstack((centroids,fission_centriods)), 'pass'
    else: # contain a single data point 
        return betas, 'error'

def fussion(centroids, MFO_index):
    """
    Merge the two centers with many-fit-one association
    """
    fussion_centroid = (centroids[MFO_index[0]] + centroids[MFO_index[1]])/2
    centroids = np.delete(centroids, MFO_index, axis=0) 
    return np.vstack((centroids, fussion_centroid)) 


def FissionFusion(X, K, seed, ofm_method='std', 
                  mfo_method='distance', init_method='random', 
                  radius_method='min', maxIter=50, start_epsilon=0.1,
                  ofm_adapt_shape=False, mfo_adapt_shape=False):
    """
    Proposed Fission Fusion kmeans procedure
    """
    dim = X.shape[1]
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
        if ofm_method == 'std':
            OFM_index = detect_ofm_std(X, betas, labels, adapt_shape=ofm_adapt_shape)
        elif ofm_method == 'radius':
            OFM_index = detect_ofm_radius(X, betas, labels, radius_method, start_epsilon=start_epsilon)
        elif ofm_method == 'dissimilar':
            OFM_index = Dissimilar(X, betas, labels)         
        OFM_history.append(OFM_index)

        #####step2: Fission #####   
        betas, state = fission(X, betas, labels, OFM_index)
        if state == 'error':
            print ('error')
            return store_C,sse 
        
        labels = compute_Voronoi(X, betas)
        #####step3: detect MFO #####
        if mfo_method == 'distance':
            MFO_index = detect_mfo_distance(X, labels, betas,adapt_shape=mfo_adapt_shape)
        if mfo_method == 'voronoi':
            MFO_index = detect_mfo_Voronoi(X, betas)
        MFO_history.append(MFO_index)
        #####step4 : Fusion #####
        betas = fussion(betas, MFO_index)
        
        assert len(betas) == K # invariance
        
        #####step5: k-means adjustment #####
        kms = KMeans(n_clusters=K, init=betas, random_state=seed, n_init=1).fit(X)
        betas = kms.cluster_centers_
        labels = kms.labels_
        new_sse = kms.inertia_
        if iterCount == 1:
            improvement_ratio = 1
        else:
            improvement_ratio = (sse[-1] - new_sse)/sse[-1]
        
        if improvement_ratio < 0: # objective do not decrease sufficiently 
            break
        else:
            sse.append(kms.inertia_)
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


###################################################
################ Related algorithm ################
###################################################




###################################
######  X-means(2000)    ##########
###################################
'''
(2000) X-means: Extending K-means with Efficient Estimation of the Number of Clusters.
code source: https://github.com/alex000kim/XMeans

xm = XMeans(n_init=1,random_state = i)
xm.fit(X,1)
store_C = [xm.cluster_centers_]
SSE = [xm.inertia_]
iter_num = xm.iter_
'''


"""
Implementation of XMeans algorithm based on
Pelleg, Dan, and Andrew W. Moore. "X-means: Extending K-means with Efficient Estimation of the Number of Clusters."
ICML. Vol. 1. 2000.
https://www.cs.cmu.edu/~dpelleg/download/xmeans.pdf
"""
EPS = np.finfo(float).eps
def loglikelihood(R, R_n, variance, M, K):
    """
    See Pelleg's and Moore's for more details.
    :param R: (int) size of cluster
    :param R_n: (int) size of cluster/subcluster
    :param variance: (float) maximum likelihood estimate of variance under spherical Gaussian assumption
    :param M: (float) number of features (dimensionality of the data)
    :param K: (float) number of clusters for which loglikelihood is calculated
    :return: (float) loglikelihood value
    """
    if 0 <= variance <= EPS:
        res = 0
    else:
        res = R_n * (np.log(R_n) - np.log(R) - 0.5 * (np.log(2 * np.pi) + M * np.log(variance) + 1)) + 0.5 * K
        if res == np.inf:
            res = 0
    return res

def get_additonal_k_split(K, X, clst_labels, clst_centers, n_features, K_sub, k_means_args):
    bic_before_split = np.zeros(K)
    bic_after_split = np.zeros(K)
    clst_n_params = n_features + 1
    add_k = 0
    for clst_index in range(K):
        clst_points = X[clst_labels == clst_index]
        clst_size = clst_points.shape[0]
        if clst_size <= K_sub:
            # skip this cluster if it is too small
            # i.e. cannot be split into more clusters
            continue
        clst_variance = np.sum((clst_points - clst_centers[clst_index]) ** 2) / float(clst_size - 1)
        bic_before_split[clst_index] = loglikelihood(clst_size, clst_size, clst_variance, n_features,
                                                     1) - clst_n_params / 2.0 * np.log(clst_size)
        kmeans_subclst = KMeans(n_clusters=K_sub, **k_means_args).fit(clst_points)
        subclst_labels = kmeans_subclst.labels_
        subclst_centers = kmeans_subclst.cluster_centers_
        log_likelihood = 0
        for subclst_index in range(K_sub):
            subclst_points = clst_points[subclst_labels == subclst_index]
            subclst_size = subclst_points.shape[0]
            if subclst_size <= K_sub:
                # skip this subclst_size if it is too small
                # i.e. won't be splittable into more clusters on the next iteration
                continue
            subclst_variance = np.sum((subclst_points - subclst_centers[subclst_index]) ** 2) / float(
                subclst_size - K_sub)
            log_likelihood = log_likelihood + loglikelihood(clst_size, subclst_size, subclst_variance, n_features,
                                                            K_sub)
        subclst_n_params = K_sub * clst_n_params
        bic_after_split[clst_index] = log_likelihood - subclst_n_params / 2.0 * np.log(clst_size)
        # Count number of additional clusters that need to be created based on BIC comparison
        if bic_before_split[clst_index] < bic_after_split[clst_index]:
            add_k += 1
    return add_k

class XMeans(KMeans):
    def __init__(self, kmax=50, max_iter=1000, **k_means_args):
        """
        :param kmax: maximum number of clusters that XMeans can divide the data in
        :param max_iter: maximum number of iterations for the `while` loop (hard limit)
        :param k_means_args: all other parameters supported by sklearn's KMeans algo (except `n_clusters`)
        """
        if 'n_clusters' in k_means_args:
            raise Exception("`n_clusters` is not an accepted parameter for XMeans algorithm")
        if kmax < 1:
            raise Exception("`kmax` cannot be less than 1")
                
        self.KMax = kmax
        self.max_iter = max_iter
        self.k_means_args = k_means_args

    def fit(self, X, K, y=None):
        #K = 1   #start from 1
        K_sub = 2
        K_old = K
        n_features = np.size(X, axis=1)
        stop_splitting = False
        iter_num = 0
        while not stop_splitting and iter_num < self.max_iter:
            K_old = K
            kmeans = KMeans(n_clusters=K, **self.k_means_args).fit(X)
            clst_labels = kmeans.labels_
            clst_centers = kmeans.cluster_centers_
            # Iterate through all clusters and determine if further split is necessary
            add_k = get_additonal_k_split(K, X, clst_labels, clst_centers, n_features, K_sub, self.k_means_args)
            K += add_k
            # stop splitting clusters when BIC stopped increasing or if max number of clusters in reached
            stop_splitting = K_old == K or K >= self.KMax
            iter_num = iter_num + 1
        # Run vanilla KMeans with the number of clusters determined above
        kmeans = KMeans(n_clusters=K_old, **self.k_means_args).fit(X)
        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_
        self.inertia_ = kmeans.inertia_
        self.n_clusters = K_old
        self.iter_ = iter_num



###################################
########   Force_kmeans()    ######
###################################
'''
(2006) Clustering by the K-Means Algorithm Using a Split and Merge Procedure


X = dataMat
store_C,sse = Frce_kmeans(X, 20, seed=1, kmax=5)
(1) p_list can be added if you want to oberseve every Voronoi split situation
(2) this algorithm only have one iteration
'''
def choose_p(X,kmax=5):
    '''
    kmax >= 2
    '''
    sse_list = []
    p_list = [] 
    for k in range(1,kmax):
        kmeans = KMeans(n_clusters=k, init="random",random_state = 0, algorithm='full',n_init=1).fit(X)
        labels = kmeans.labels_
        betas = kmeans.cluster_centers_
        sse = kmeans.inertia_
        sse_list.append(sse)
        if k > 1:
            p = sse_list[-1]/sse_list[-2]
            p_list.append(p)
    p_index = np.argmin(p_list) + 2  
    return p_index

def Frce_kmeans(X, k, seed=1, kmax=5):
    centroids_total = np.zeros(X.shape[1])
    kmeans = KMeans(n_clusters=k, init='random',random_state=seed, n_init=1, algorithm='full').fit(X)
    labels = kmeans.labels_
    centroids_old = kmeans.cluster_centers_
    #p_list = []
    for index in range(k):
        p_index = choose_p(X[labels == index], kmax=5) 
        #p_list.append(p_index)
        kmeans_sub = KMeans(n_clusters=p_index).fit(X[labels == index])
        center = kmeans_sub.cluster_centers_
        centroids_total = np.vstack((centroids_total,center))
    centroids_total = np.delete(centroids_total, 0, axis=0) 
    centroids_split = centroids_total.copy()
    for i in range(len(centroids_total)-k):
        MFO_index = detec_mfo_dist(centroids_total)
        centroids_total = merge(centroids_total,MFO_index)
    
    sklkms=KMeans(n_clusters=k, init=centroids_total, n_init=1, algorithm='full').fit(X)
    store_C = [sklkms.cluster_centers_] 
    sse = [sklkms.inertia_]
    return store_C, sse#, p_list

# merge criterion
def detec_mfo_dist(betas):
    """ output the indices of a pair of clusters whose distance is the smallest"""
    k = len(betas)
    min_dist = np.inf
    for i in range(k-1):
        for j in range(i+1,k):
            distance = np.sum((betas[i,:] - betas[j,:])**2)
            if distance < min_dist:
                MFO_index = [i,j]
                min_dist = distance
    return MFO_index

def merge(centroids, MFO_index):
    """
    Merge the two centers with many-fit-one association
    """
    fussion_centroid = (centroids[MFO_index[0]] + centroids[MFO_index[1]])/2
    centroids = np.delete(centroids, MFO_index, axis=0) 
    return np.vstack((centroids, fussion_centroid))


################################
######    Dissimilar  ##########
################################
'''
(2016) Robust K-means algorithm with automatically splitting and merging clusters and its applications for surveillance data
inital with K, but according to the paper, this k is not robust, it should be adjust throught this aglrothm
'''
def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

#d_bar: formula(5) of paper
def D_bar(centroids):
    A = len(centroids)
    total_sum = 0
    for i in range(A):
        for j in range(A):
            total_sum += distEclud(centroids[i,:], centroids[j,:])**2
    return total_sum/(A*(A-1)) # pairwise not including self

#D_intra: formula(9) of paper
def D_intra(X, centroid):
    dist = []
    for i in range(len(X)):
        dist.append(distEclud(X[i,:],centroid)**2)
    return max(dist)+min(dist)

def compute_equivalent_class(record):
    """output the equivalent classes from pairwise relation 
       [[1,2],[2,3],[4,5]] --> [[1,2,3],[4,5]]"""
    equivalent_class = {}
    class_members=[]
    max_class_number = -1
    for pair in record:
        if (pair[0] in equivalent_class) and (not (pair[1] in equivalent_class)):
            equivalent_class[pair[1]] = equivalent_class[pair[0]]
        if (not(pair[0] in equivalent_class)) and (not (pair[1] in equivalent_class)):
            max_class_number+=1
            equivalent_class[pair[0]] = max_class_number
            equivalent_class[pair[1]] = max_class_number
    for c in range(max_class_number+1):
        class_members.append([index for index,val in equivalent_class.items() if val==c])
    return class_members
            
def batch_fussion(betas, class_members):
    """
    merge centers in the same equivalent class
    """
    merge_centers = None
    all_merge_indices = []
    for c, merge_indices in enumerate(class_members):
        all_merge_indices += merge_indices
        if isinstance(merge_centers,np.ndarray):
            merge_centers = np.vstack((merge_centers,np.mean(betas[merge_indices],axis=0)))
        else:
            merge_centers = np.mean(betas[merge_indices],axis=0)
    betas = np.delete(betas, all_merge_indices, axis=0)
    betas = np.vstack((betas, merge_centers))
    return betas

def Ds_kmeans(X, k, seed=1, batch_merge=False):
    #kmeans
    kmeans = KMeans(n_clusters=k, init='random',random_state=seed, n_init=1, algorithm='full').fit(X)
    labels = kmeans.labels_
    betas = kmeans.cluster_centers_
    split_state = True
    merge_state = True
    k_observe = []
    #----------------split------------------------
    while split_state:
        split_state = False
        d_bar = D_bar(betas)
        d_list = []
        d_inx = [] #store split index
        for index in range(len(betas)):
            d_intra = D_intra(X[labels == index],betas[index,:])
            #print (d_intra, d_bar/20)
            if d_intra > (d_bar/20):
                split_state =True
                d_list.append(d_intra)
                d_inx.append(index)
        betas = np.delete(betas, d_inx, axis=0) 
        for index in d_inx:
            fis_kms=KMeans(n_clusters=2,init='random',algorithm='full',random_state=0,n_init=1).fit(X[labels == index])
            fis_betas = fis_kms.cluster_centers_      
            betas=np.vstack((betas,fis_betas)) 
        #print (len(betas))
        # update labels 
        labels = compute_Voronoi(X, betas)
    k_observe.append(len(betas))
    #----------------merge------------------------
    next_k = 0
    iter1=0
    record = []
    while merge_state and len(betas)>2:
        merge_state = False
        d_bar = D_bar(betas)
        min_d = np.inf
        for i in range(len(betas)-1):
            for j in range(i+1,len(betas)):
                d_inter = distEclud(betas[i,:], betas[j,:])**2
                if d_inter < d_bar/40:
                    merge_state = True
                    if batch_merge:
                        record.append([i,j])
                    else:
                        if min_d > d_inter:
                            min_d = d_inter
                            record = [i,j] 
                             
        if merge_state:
            if batch_merge:
                class_index = compute_equivalent_class(record)
                betas = batch_fussion(betas, class_index)
            else:
                betas = fussion(betas, record)
            #fussion_centroid = (betas[record[0]] + betas[record[1]])/2
            #betas = np.delete(betas, record, axis=0)
            #betas = np.vstack((betas, fussion_centroid)) 
            iter1 +=1
    #print (len(betas))
    k_observe.append(len(betas))
       
    kms=KMeans(n_clusters=len(betas), init=betas, n_init=1, algorithm='full').fit(X)   
    store_C = [kms.cluster_centers_] 
    sse = [kms.inertia_]
            
    return store_C,sse,k_observe
    
