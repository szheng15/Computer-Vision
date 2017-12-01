import numpy as np
from scipy import randn, misc
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
import os
import random

'''
0. Import SciPy clustering package for the k-means clustering
'''

def image_kmeans(image, k, filename):
    h,w,d = image.shape
    image_float = np.array(image, dtype=np.float64) / 255
    image_array = np.reshape(image_float, (h * w, d))
    centroids, _ = kmeans(image_array, k)
    idx,_ = vq(image_array, centroids)
    codebook_random = random.sample(image_array, k)
    for row in range(h):
        for col in range(w):
            image_float[row, col] = [x * 255 for x in codebook_random[idx[row * w  + col]]]
    im = image_float.astype(np.uint8)
    plt.figure()
    plt.imshow(im)
    misc.imsave('output/' + filename + '.jpg', im)

def texture_segmentation(filter_response, k, save_texture=False, filename=''):
    h,w = filter_response[0].shape
    image_array = np.zeros((h * w, 8), dtype=np.float64)
    image_float = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(8):
        image_array[:,i] = np.reshape(filter_response[i], h*w)
    centroids, _ = kmeans(image_array, k)
    idx,_ = vq(image_array, centroids)
    codebook_random = [[random.random() for j in range(3)] for i in range(k)]
    for row in range(h):
        for col in range(w):
            image_float[row, col] = np.asarray(codebook_random[idx[row * w  + col]]) * 255
    im = image_float.astype(np.uint8)
    plt.figure()
    plt.imshow(im, cmap=plt.cm.gray)
    if save_texture:
        misc.imsave('output/' + filename + '.jpg', im)

def main():
    '''
    Problem 1: K-means clustering
    '''
    '''
    1. Generate three sets of random points
    '''
    # generate a set of 100 random points in 2-D space (class 1), following a normal distribution
    # Multiply their 2-D coordinates by 2.
    class1 = 2.0 * randn(100, 2)
    # generate another set of 100 random points (class 2), whose mean is at [5,5]
    class2 = randn(100, 2) + np.array([5, 5])
    # generate the third set of 50 random points (class 3), following a standard Gaussian distribution with mean[5,0]
    class3 = randn(100, 2) + np.array([5, 0])

    '''
    2. Do k-means (k = 3) and visualize results
    '''
    data = np.vstack((class1, class2, class3))
    # computing K-Means with K = 3 (3 clusters)
    centroids, _ = kmeans(data, 3)
    # assign each sample to a cluster
    idx, _ = vq(data, centroids)
    # some plotting using numpy's logical indexing

    plt.figure()
    plt.plot(data[idx == 0, 0], data[idx == 0, 1], 'ob', data[idx == 1, 0], data[idx == 1, 1], 'or', data[idx == 2, 0],
             data[idx == 2, 1], 'og')
    plt.plot(centroids[:, 0], centroids[:, 1], 'sk', markersize=8)
    # save figure as 2d_clustering.jpg
    if not os.path.exists('output'):
        os.makedirs('output')

    plt.savefig('output/2d_clustering.jpg')

    '''
    Problem 2: Color-based image segmentation using K-means clustering
    '''
    # 0. download zebra.jpg and fish.jpg
    # 1. Treat each pixel in the test image as a 3-D point based on its RGB values
    zebra_image = misc.imread('images/zebra.jpg')
    fish_image = misc.imread('images/fish.jpg')
    # 2. Cluster them using k-means (try k=4,8,12, ...)
    image_kmeans(zebra_image, 12, 'zebra_color_clustering')
    image_kmeans(fish_image, 8, 'fish_color_clustering')

    '''
    Problem 3: Texture-based image segmentation using K-means clustering
    '''
    zebra_image = misc.imread('images/zebra.jpg', mode='F')
    fish_image = misc.imread('images/fish.jpg', mode='F')
    import pa2

    filter_bank = pa2.create_filter_bank()
    filter_response = pa2.convolve_filter(zebra_image, filter_bank)
    texture_segmentation(filter_response, 15, True, 'zebra_texture_clustering')
    filter_response = pa2.convolve_filter(fish_image, filter_bank)
    texture_segmentation(filter_response, 5, True, 'fish_texture_clustering')

    plt.show()
if __name__ == "__main__":
    main()
