import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def L2norm(f1, f2):
    '''
    Problem 1:
    2. Implement a function to measure "Euclidean distance" between two features (i.e., L2 norm)
    '''
    return np.linalg.norm(f1 - f2)

def main():
    '''
    main funtion
    '''
    img1 = cv2.imread('images/box.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/box_in_scene.png', cv2.IMREAD_GRAYSCALE)
    if not os.path.exists('output'):
        os.makedirs('output')

    '''
    Problem 1: SIFT local feature extraction and matching
    '''
    '''
    1. Extract SIFT features from the images using OpenCV
    Save the results as "box_sift.jpg" and "box_in_scene_sift.jpg"
    '''
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    img = cv2.drawKeypoints(img1, kp1, None)
    cv2.imwrite('output/box_sift.jpg', img)
    kp2, des2 = sift.detectAndCompute(img2, None)
    img = cv2.drawKeypoints(img2, kp2, None)
    cv2.imwrite('output/box_in_scene_sift.jpg', img)

    '''
    3. Use the implemented function to measure the pairwise distance between each feature from
    box.png and each feature from box_in_scene.png
    '''
    m1 = des1.shape[0]
    m2 = des2.shape[1]
    distance = np.zeros((m1, m2))
    for x1 in range(m1):
        for x2 in range(m2):
            distance[x1, x2] = L2norm(des1[x1], des2[x2])
    '''
    4. Based on the distance matrix, for each feature in box.png, find the corresponding feature in
    box_in_scene.png. Visualize the matching results with lines. Only display 50 best matches
    '''
    distp = distance.reshape(m1 * m2)
    min50idx = distp.argsort()
    smp1 = min50idx / m2 # match point in image 1
    smp2 = min50idx % m2 # match point in image 2
    # find matches (not duplicated features)
    mp1 = []
    mp2 = []
    count = 0
    ip = 0
    while count < 50:
        if not(smp1[ip] in mp1) and not(smp2[ip] in mp2):
            mp1.append(smp1[ip])
            mp2.append(smp2[ip])
            count += 1
        ip += 1

    '''
    Visualize the matching results with lines
    '''
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    out_image = np.zeros((max(h1, h2), w1 + w2))
    out_image[0:h1, 0:w1] = img1
    out_image[0:h2, w1:] = img2
    plt.figure()
    plt.imshow(out_image)
    for i in range(len(mp1)):
        ip1 = mp1[i]
        ip2 = mp2[i]
        plt.plot((kp1[ip1].pt[0], kp2[ip2].pt[0] + w1) , (kp1[ip1].pt[1], kp2[ip2].pt[1]), 'k-')
    plt.savefig('output/matching.jpg')

    '''
    5. Use BFMatcher of OpenCV to do the same matching
    '''
    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
    plt.figure()
    plt.imshow(img3)
    plt.savefig('output/matching2.jpg')

    '''
    Problem 2: SIFT matching spatial verification
    '''
    #match_RANSAC(kp1, des1, kp2, des2, 'output/box_match_RANSAC.jpg')
    '''
    1. Use the OpenCV's cv2.findHomography function to find the homography between SIFT features
    in one image (box.png) to SIFT features in the other image (box_in_scene.png) using RANSAC
    '''
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    '''
    2. Visualize the bounding box of the object in box_in_scene.png using the homography matrix.
    Also visualize the valid SIFT matches yourself
    '''
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    plt.figure()
    plt.imshow(img3, 'gray')
    plt.savefig('output/box_match_RANSAC.jpg')
    plt.show()

if __name__ == "__main__":
    main()
