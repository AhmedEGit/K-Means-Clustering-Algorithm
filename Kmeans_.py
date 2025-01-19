import numpy as np
import matplotlib.pyplot as plt
import itertools

def show_clusters(x, cluster_labels, centroids, title=None):
    """
    Create plot of feature vectors with same colour for members of same cluster.

    :param x:               feature vectors, np array (dim, number_of_vectors) (float64/double),
                            where dim is arbitrary feature vector dimension
    :param cluster_labels:  cluster index for each feature vector, np array (number_of_vectors, ),
                            array contains only values from 1 to k,
                            i.e. cluster_labels[i] is the index of a cluster which the vector x[:,i] belongs to.
    :param centroids:       cluster centers, np array (dim, k) (float64/double),
                            i.e. centroids[:,i] is the center of the i-th cluster.
    """

    cluster_labels = cluster_labels.flatten()
    clusters = np.unique(cluster_labels)
    markers = itertools.cycle(['*','o','+','x','v','^','<','>'])

    plt.figure(figsize=(8,7))
    for i in clusters:
        cluster_x = x[:, cluster_labels == i]
        plt.plot(cluster_x[0], cluster_x[1], next(markers))
    plt.axis('equal')

    len = centroids.shape[1]
    for i in range(len):
        plt.plot(centroids[0, i], centroids[1, i], 'm+', ms=10, mew=2)

    plt.axis('equal')
    plt.grid('on')
    if title is not None:
        plt.title(title)

def k_means(x, k, max_iter, show=False, init_means=None):
    """
    Implementation of the k-means clustering algorithm.

    :param x:               feature vectors, np array (dim, number_of_vectors)
    :param k:               required number of clusters, scalar
    :param max_iter:        stopping criterion: max. number of iterations
    :param show:            (optional) boolean switch to turn on/off visualization of partial results
    :param init_means:      (optional) initial cluster prototypes, np array (dim, k)

    :return cluster_labels: cluster index for each feature vector, np array (number_of_vectors, )
                            array contains only values from 0 to k-1,
                            i.e. cluster_labels[i] is the index of a cluster which the vector x[:,i] belongs to.
    :return centroids:      cluster centroids, np array (dim, k), same type as x
                            i.e. centroids[:,i] is the center of the i-th cluster.
    :return sq_dists:       squared distances to the nearest centroid for each feature vector,
                            np array (number_of_vectors, )

    Note : The iterative procedure terminates if either maximum number of iterations is reached
            or there is no change in assignment of data to the clusters.

    """
    # Number of vectors
    n_vectors = x.shape[1]
    cluster_labels = np.zeros([n_vectors], np.int32)
    sq_dists = np.zeros(n_vectors)
    # Means initialization
    if init_means is None:
        ind = np.random.choice(n_vectors, k, replace=False)
        centroids = x[:, ind]
    else:
        centroids = init_means

    i_iter = 0


    while i_iter < max_iter:
        previous_centroids = np.copy(cluster_labels)
        for i in range(n_vectors):
            c = np.array([np.linalg.norm(x.T[:][i]-centroids.T,axis=1)**2])
            cluster_labels[i] = np.argmin(c)
            sq_dists[i] = np.min(c)

        ind = np.random.choice(n_vectors, k, replace=False)
        for j in range(centroids.shape[1]):

            if np.sum([cluster_labels == j]) > 0:

                centroids.T[:][j] = np.mean(x.T[:][cluster_labels == j],axis=0)
            else:
                if init_means is None:

                    centroids.T[:,j] = x[:, ind[j]]
                else:
                    centroids.T[:,j] = init_means[:,j]
        if (cluster_labels == previous_centroids).all():
            return cluster_labels, centroids, sq_dists

        i_iter += 1    # incrementing iterations

        # Ploting partial results
        if show:
            print('Iteration: {:d}'.format(i_iter))
            show_clusters(x, cluster_labels, centroids, title='Iteration: {:d}'.format(i_iter))

    if show:
        print('Done.')


    return cluster_labels, centroids, sq_dists




def compute_measurements(images):
    """
    computes 2D features from image measurements

    :param images: array of images, np array (H, W, N_images) (np.uint8)
    :return x:     array of features, np array (2, N_images)
    """

    images = images.astype(np.float64)
    H, W, N = images.shape

    left = images[:, :(W//2), :]
    right = images[:, (W//2):, :]
    up = images[:(H//2), ...]
    down = images[(H//2):, ...]

    L = np.sum(left, axis=(0, 1))
    R = np.sum(right, axis=(0, 1))
    U = np.sum(up, axis=(0, 1))
    D = np.sum(down, axis=(0, 1))

    a = L - R
    b = U - D

    x = np.vstack((a, b))
    return x


def k_means_multiple_trials(x, k, n_trials, max_iter, show=False):
    """
    Performs several trials of the k-centroids clustering algorithm in order to
    avoid local minima. Result of the trial with the lowest "within-cluster
    sum of squares" is selected as the best one and returned.

    :param x:               feature vectors, np array (dim, number_of_vectors)
    :param k:               required number of clusters, scalar
    :param n_trials:        number of trials, scalars
    :param max_iter:        stopping criterion: max. number of iterations
    :param show:            (optional) boolean switch to turn on/off visualization of partial results

    :return cluster_labels: cluster index for each feature vector, np array (number_of_vectors, ),
                            array contains only values from 0 to k-1,
                            i.e. cluster_labels[i] is the index of a cluster which the vector x[:,i] belongs to.
    :return centroids:      cluster centroids, np array (dim, k), same type as x
                            i.e. centroids[:,i] is the center of the i-th cluster.
    :return sq_dists:       squared distances to the nearest centroid for each feature vector,
                            np array (number_of_vectors, )
    """
    opt1 = []
    opt2 = []

    opt3 = []
    l = np.array([])
    for i in range(n_trials):
        a = k_means(x,k,max_iter,show,init_means=None)

        opt1.append(a[0])        # append cluster labels
        opt2.append(a[1])       # append centroids

        opt3.append(a[2])         # append squared distance
        wcss = np.sum(a[2])             # calculate wcss (within-cluster sum of squares)
        l = np.append(l,wcss)

    c = np.argmin(l)
    cluster_labels, centroids, sq_dists = opt1[c],opt2[c],opt3[c]

    return cluster_labels.astype('int32'), centroids, sq_dists


def random_sample(weights):
    """
    picks randomly a sample based on the sample weights.

    :param weights: array of sample weights, np array (n, )
    :return idx:    index of chosen sample, scalar

    Note: use np.random.uniform() for random number generation in open interval (0, 1)
    """
    a = weights.copy()*(1/(np.sum(weights.copy())))


    s = np.cumsum(a)
    b = np.random.uniform(0,1)


    for j in range(len(s)):
        if s[j]<b and b<s[j+1]:
            return j+1

    return 0




def k_meanspp(x, k):
    """
    performs k-means++ initialization for k-means clustering.

    :param x:           Feature vectors, np array (dim, number_of_vectors)
    :param k:           Required number of clusters, scalar

    :return centroids:  proposed centroids for k-means initialization, np array (dim, k)
    """
    col = random_sample(np.ones(x.shape[1]))
    first_cluster = x[:,col]
    centroids = first_cluster

    for _ in range(1,k):
        min_dist = np.array([])

        for i in range(x.shape[1]):
            a = np.linalg.norm((x[:,i].T-centroids).T,None,axis=0)   # calculate the distance between the i-th vector of x and centres

            min_dist = np.append(min_dist,np.min(a))     # we take the lowest distance
        dist_sq = min_dist**2
        weights = dist_sq
        ind = random_sample(weights)
        centroids = np.vstack((centroids,x[:,ind]))


    return centroids.T




def quantize_colors(im, k):
    """
    Image color quantization using the k-means clustering. A subset of 1000 pixels
    is first clustered into k clusters based on their RGB color.
    Quantized image is constructed by replacing each pixel color by its cluster centroid.

    :param im:          image for quantization, np array (h, w, 3) (np.uint8)
    :param k:           required number of quantized colors, scalar
    :return im_q:       image with quantized colors, np array (h, w, 3) (uint8)

    note: make sure that the k-means is run on floating point inputs.
    """
    (h,w,c) = im.shape
    image_2D = np.reshape(im,(h*w,c)).astype(np.float64)   # reshaping the image : each row is a data point

    inds = np.random.randint(0, (h*w) - 1, 1000)
    rando_pixels = image_2D[inds,:]  # 1000 data points selected randomly
    tr = np.transpose(rando_pixels)
    clustering_1000pix = k_means(tr,k,float('inf'),show=False,init_means=None)
    centroids = clustering_1000pix[1]   # centroids with (3,8) shape


    cop = np.copy(image_2D)
    for i in range(image_2D.shape[0]):  # for each data point
        dist = np.linalg.norm(cop[i,:]-centroids.T,None,axis=1)  # calculate the distance between a data point and centers

        ind_center = np.argmin(dist)   # the indice of the center having the minimum distance with the data point
        cop[i,:] = centroids[:,ind_center]   

    im_q = np.reshape(cop,(h,w,c)).astype(np.uint8)

    return im_q


def gen_kmeanspp_data(mu=None, sigma=None, n=None):
    """
    generates data with n_clusterss normally distributed clusters

    It generates 4 clusters with 80 points by default.

    :param mu:          mean of normal distribution, np array (dim, n_clusters)
    :param sigma:       std of normal distribution, scalar
    :param n:           number of output points for each distribution, scalar
    :return samples:    dim-dimensional samples with n samples per cluster, np array (dim, n_clusters * n)
    """

    sigma = 1. if sigma is None else sigma
    mu = np.array([[-5, 0], [5, 0], [0, -5], [0, 5]]) if mu is None else mu
    n = 80 if n is None else n

    samples = np.random.normal(np.tile(mu, (n, 1)).T, sigma)
    return samples
