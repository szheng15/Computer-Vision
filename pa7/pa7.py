import cv2
import numpy as np
from scipy.spatial.distance import *
from scipy.cluster.vq import kmeans, vq
import os

def main():
    '''
    Problem 2: Image retrieval using Bag-of-words
    '''
    categories = ['airplanes', 'chair', 'crocodile', 'headphone', 'soccer_ball', 'camera',
                  'crab', 'elephant', 'pizza', 'starfish']

    deslist = []
    train_images_per_category = 32
    num_categories = len(categories)
    num_train_images = num_categories * train_images_per_category
    k = 200
    histfile = 'output/histogram_pa7.npy'
    centroidsfile = 'output/centroids_pa7.npy'
    '''
    1. For each image, extract SIFT features.
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    for category in categories:
        for i in range(train_images_per_category):
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
            des_train = np.vstack([des_train, deslist[i * train_images_per_category]])
        centroids, _ = kmeans(des_train, k)
        np.save(centroidsfile, centroids)

    '''
    3. Based on the learned cluster centers, for each of image, construct a
    bag-of-visual-words histogram
    '''
    if os.path.isfile(histfile):
        histogramlist = np.load(histfile)
    else:
        histogramlist = np.zeros((num_train_images, k), np.float32)
        for i in range(num_train_images):
            idx,_ = vq(deslist[i], centroids)
            hist, _ = np.histogram(idx, k, (0, k))
            histogramlist[i] = hist
        np.save(histfile, histogramlist)

    '''
    4.  Classify each test image using k-nearest neighbor classifier. For each test image,
    find the k=(3 and 5) training images having the smallest Euclidean histogram distance.
    Decide the object class of the test image based on ground truth labels of those 3 (or 5) training images.
    Report the classification accuracy of each class
    '''
    for cidx in range(len(categories)):
        category = categories[cidx]
        imgdir = 'images/caltec101/' + category
        total_test_images_per_class = 0
        true_positives_per_class_3 = 0
        true_positives_per_class_5 = 0
        for file in os.listdir(imgdir):
            if file.endswith(".jpg"):
                fileidx = int(file[6:10])
                if fileidx > train_images_per_category:
                    imgname = imgdir + '/' + file
                    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
                    _, des = sift.detectAndCompute(img, None)
                    idx,_ = vq(des, centroids)
                    hist, _ = np.histogram(idx, k, (0, k))
                    dist = []
                    for i in range(num_train_images):
                        dist.append(euclidean(hist, histogramlist[i]))

                    sortidx = np.asarray(dist).argsort()[:5]
                    found_3 = False
                    found_5 = False
                    for i in range(3):
                        catidx = sortidx[i] // train_images_per_category
                        if cidx == catidx:
                            found_3 = True
                            break
                    for i in range(5):
                        catidx = sortidx[i] // train_images_per_category
                        if cidx == catidx:
                            found_5 = True
                            break
                    if found_3:
                        true_positives_per_class_3 += 1
                    if found_5:
                        true_positives_per_class_5 += 1
                    total_test_images_per_class += 1
        print('Top-3 classification accuracy of {}: {:6.2f}%'.format(category, float(true_positives_per_class_3) / total_test_images_per_class * 100))
        print('Top-5 classification accuracy of {}: {:6.2f}%'.format(category, float(true_positives_per_class_5) / total_test_images_per_class * 100))

if __name__ == "__main__":
    main()