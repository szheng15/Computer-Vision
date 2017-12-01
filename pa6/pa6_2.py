import cv2
import numpy as np
from scipy.spatial.distance import *
from scipy.cluster.vq import kmeans, vq, whiten
import matplotlib.pyplot as plt
import os
import random
from shutil import copyfile

def main():
    '''
    Problem 2: Image retrieval using Bag-of-words
    '''
    categories = ['airplanes', 'chair', 'crocodile', 'headphone', 'soccer_ball', 'camera',
                  'crab', 'elephant', 'pizza', 'starfish']

    deslist = []
    images_per_category = 10
    num_categories = len(categories)
    num_images = num_categories * images_per_category
    k = 200
    histfile = 'output/histogram.npy'
    centroidsfile = 'output/centroids.npy'
    '''
    1. For each image, extract SIFT features.
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    for category in categories:
        for i in range(images_per_category):
            imgname = 'images/caltec101/' + category + '/image_' + '{:0>4d}'.format(i + 1) + '.jpg'
            img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
            _, des = sift.detectAndCompute(img, None)
            deslist.append(des)
    if not os.path.exists('output'):
        os.makedirs('output')

    '''
    2. Use k-means to find cluster centers (k=200) of 128-D SIFT descriptors. When doing this,
    only use SIFT features from "image_0001.jpg" of each category (i.e., we only use 10 images).
    '''
    if os.path.isfile(centroidsfile):
        centroids = np.load(centroidsfile)
    else:
        des_train = np.zeros((0, 128))
        for i in range(num_categories):
            des_train = np.vstack([des_train, deslist[i * images_per_category]])
        centroids, _ = kmeans(des_train, k)
        np.save(centroidsfile, centroids)

    '''
    Based on the learned cluster centers, for each of the 100 images, construct a
    bag-of-visual-words histogram
    '''
    if os.path.isfile(histfile):
        histogramlist = np.load(histfile)
    else:
        histogramlist = np.zeros((num_images, k), np.float32)
        for i in range(num_images):
            idx,_ = vq(deslist[i], centroids)
            hist, _ = np.histogram(idx, k, (0, k))
            histogramlist[i] = hist
        np.save(histfile, histogramlist)

    '''
    4. Use "image_0001.jpg" of each category as the query image. Try to find 5 best matches per
    query. The match score is computed as the Euclidean distance between two BoW histograms:
    the lower distance implies the higher match score
    '''
    for category in categories:
        imgname = 'images/caltec101/' + category + '/image_0001.jpg'
        img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
        _, des = sift.detectAndCompute(img, None)
        idx,_ = vq(des, centroids)
        hist, _ = np.histogram(idx, k, (0, k))
        dist = []
        '''
        Use scipy.spatial.distance.euclidean to compute Euclidean distance.
        '''
        for i in range(num_images):
            dist.append(euclidean(hist, histogramlist[i]))

        sortidx = np.asarray(dist).argsort()[:5]
        for i in range(5):
            catidx = sortidx[i] / images_per_category
            imgidx = sortidx[i] % images_per_category
            catname = categories[catidx]
            copyfile('images/caltec101/' + catname + '/image_'+ '{:0>4d}'.format(imgidx + 1) + '.jpg',
                     'output/' + category + '_query_result' + str(i + 1) + '.jpg')

if __name__ == "__main__":
    main()