import numpy as np
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

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


############################################
######    Algorithm8: Dissimilar  ##########
############################################
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
    
