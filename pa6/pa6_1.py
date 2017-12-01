import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq, whiten
import matplotlib.pyplot as plt
import os
import random

def match_RANSAC(img1, kp1, des1, img2, kp2, des2, filename):
    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    cv2.imwrite(filename, img3)
    plt.figure()
    plt.imshow(img3, 'gray')

def main():
    '''
    Problem 1: Bag-of-words matching with SIFT
    '''

    '''
    1. Extract SIFT features from the images using OpenCV.
    '''
    img1 = cv2.imread('images/elephant_model.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/staple_remover_model.png', cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread('images/cluttered_desk.png', cv2.IMREAD_GRAYSCALE)
    if not os.path.exists('output'):
        os.makedirs('output')

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    kp3, des3 = sift.detectAndCompute(img3, None)

    '''
    2. Implement bag-of-words. Concatenate all features extracted from all three images.
    Use kmeans library to cluster them into k clusters
    '''
    des = np.concatenate((des1, des2, des3))
    des = whiten(des)
    k = 25
    centroids, _ = kmeans(des, k)
    idx,_ = vq(des, centroids)
    codebook_random = [[int(random.random() * 255) for j in range(3)] for i in range(k)]

    '''
    Visualize the clustering results on top of each image by coloring each SIFT feature based on
    its cluster ID.
    Save the results as elephant_model_bow.jpg, staple_remover_model_bow.jpg, and cluttered_desk_bow.jpg
    '''
    offset2 = len(kp1)
    offset3 = len(kp1) + len(kp2)

    tmpimg = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for i in range(len(kp1)):
        cv2.circle(tmpimg, (int(kp1[i].pt[0]), int(kp1[i].pt[1])), 3, codebook_random[idx[i]])
    cv2.imwrite('output/elephant_model_bow.jpg', tmpimg)
    plt.figure()
    plt.imshow(tmpimg)

    tmpimg = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for i in range(len(kp2)):
        cv2.circle(tmpimg, (int(kp2[i].pt[0]), int(kp2[i].pt[1])), 3, codebook_random[idx[i + offset2]])
    cv2.imwrite('output/staple_remover_model_bow.jpg', tmpimg)
    plt.figure()
    plt.imshow(tmpimg)

    tmpimg = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
    for i in range(len(kp3)):
        cv2.circle(tmpimg, (int(kp3[i].pt[0]), int(kp3[i].pt[1])), 3, codebook_random[idx[i + offset3]])
    cv2.imwrite('output/cluttered_desk_bow.jpg', tmpimg)
    plt.figure()
    plt.imshow(tmpimg)

    '''
    3. Match SIFT features
    (1) between elephant_model.png and cluttered_desk.png and
    (2) staple_remover_model.png and cluttered_desk.png, solely based on their cluster ids.
    '''
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    h3, w3 = img3.shape

    out1 = np.zeros((max(h1, h3), w1 + w3), np.uint8)
    out1[0:h1, 0:w1] = img1
    out1[0:h3, w1:] = img3
    for i in range(len(kp1)):
        for j in range(len(kp3)):
            if idx[i] == idx[j + offset3]:
                cv2.line(out1, (int(kp1[i].pt[0]), int(kp1[i].pt[1])), (int(kp3[j].pt[0] + w1), int(kp3[j].pt[1])), (0, 0, 0))
    cv2.imwrite('output/match1.jpg', out1)
    plt.figure()
    plt.imshow(out1, cmap=plt.cm.gray)

    out2 = np.zeros((max(h2, h3), w2 + w3), np.uint8)
    out2[0:h2, 0:w2] = img2
    out2[0:h3, w2:] = img3
    for i in range(len(kp2)):
        for j in range(len(kp3)):
            if idx[i + offset2] == idx[j + offset3]:
                cv2.line(out2, (int(kp2[i].pt[0]), int(kp2[i].pt[1])), (int(kp3[j].pt[0] + w2), int(kp3[j].pt[1])), (0, 0, 0))
    cv2.imwrite('output/match2.jpg', out2)
    plt.figure()
    plt.imshow(out2, cmap=plt.cm.gray)

    '''
    4. Apply RANSAC
    '''
    match_RANSAC(img1, kp1, des1, img3, kp3, des3, 'output/match1_RANSAC.jpg')
    match_RANSAC(img2, kp2, des2, img3, kp3, des3, 'output/match2_RANSAC.jpg')

    plt.show()

if __name__ == "__main__":
    main()
