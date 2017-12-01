import cv2
import numpy as np
import os
import zipfile
from scipy.spatial.distance import *
from scipy.cluster.vq import kmeans, vq
import random

def calcOpticalFlow(directory, zipfilename):
    prev_img = None
    zipf = zipfile.ZipFile(zipfilename, 'w')
    tmpfile = 'output/tmp.png'
    for file in sorted(os.listdir(directory)):
        if not file.endswith('.png'):
            continue
        img = cv2.imread(directory + '/' + file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if prev_img is None:
            prev_img = img_gray

        flow = np.zeros((img_gray.shape[0], img_gray.shape[1], 2), np.float32)
        cv2.calcOpticalFlowFarneback(prev_img, img_gray, flow, 0.5, 1, 12, 2, 5, 1.1, 0)
        for y in range(0, img.shape[0], 5):
            for x in range(0, img.shape[1], 5):
                floatxy = flow[y, x] * 3
                cv2.line(img, (x, y), (int(x + floatxy[0]), int(y + floatxy[1])), (255, 0, 0))
                #cv2.circle(img, (x, y), 1, (0, 0, 0), -1)
        cv2.imwrite(tmpfile, img)
        zipf.write(tmpfile, file)
        os.remove(tmpfile)
        prev_img = img_gray

    zipf.close()

def xyt(videofile):
    '''
    2-1. Load each video as a 3-D array (XYT data)
    :param videofile:
    :return: label, 3-D array
    '''
    label = int(os.path.basename(videofile).split('_')[1].split('.')[0])
    cap = cv2.VideoCapture(videofile)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    return label, buf[0:fc]

def getHOGperFrame(frame):
    '''
    2-2. Prepare a function hist = GetHOGperFrame(image), which extracts histogram of gradients
    (HOG) descriptor from each image frame
    '''
    '''
    divide the frame into 5-by-5 spatial regions, and then count the number of pixels
    (in each region) belonging to each of 9 gradient orientation bins
    '''
    nrow = 5
    ncol = 5
    bin_n = 9
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cells = np.asarray([np.hsplit(row, nrow) for row in np.vsplit(img_gray, ncol)])
    hists = np.asarray([])
    for row in range(nrow):
        for col in range(ncol):
            cellimg = cells[row, col]
            '''
            For each spatial region, iterate through each pixel in the bin,
            get its gradient orientation, and assign it to one of the 9 bins
            based on the orientation value
            '''
            gx = cv2.Sobel(cellimg, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(cellimg, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            '''
            We will use unsigned gradient: i.e., 20 degrees and 200 degrees (180+20 degrees)
            are treated identically
            '''
            ang = np.mod(ang, 180)
            # quantizing binvalues in (0...8)
            bins = np.int32(ang * bin_n / 180)
            hist = np.bincount(bins.ravel(), minlength=bin_n)
            '''
            Concatenate all of them, obtaining a total of 225 bins (i.e., an array with 225 values)
            '''
            hists = np.append(hists, hist)
    return hists

def draw_hog(frame, filename):
    hog = getHOGperFrame(frame)
    rows = frame.shape[0]
    cols = frame.shape[1]
    nrow = 5
    ncol = 5
    bin_n = 9
    length = 24.0
    img = np.zeros((rows, cols, 1), np.uint8)
    for row in range(nrow):
        for col in range(ncol):
            yc = int(rows / nrow * (row + 0.5))
            xc = int(cols / ncol * (col + 0.5))
            startpos = bin_n * (row * ncol + col)
            hist = hog[startpos:startpos+bin_n]
            for i in range(bin_n):
                angle = float(i * 180 / bin_n)
                dx, dy = cv2.polarToCart(np.array([length]), np.array([angle]), angleInDegrees=True)
                b = hist[i] / 3456 * 255
                cv2.line(img, (xc+dx[0], yc+dy[0]), (xc-dx[0], yc-dy[0]), (b, b, b))
    cv2.imwrite(filename, img)

def getHOFperFrame(oldframe, frame):
    '''
    2-2. Prepare a function hist = GetHOGperFrame(image), which extracts histogram of gradients
    (HOG) descriptor from each image frame
    '''
    '''
    divide the frame into 5-by-5 spatial regions, and then count the number of pixels
    (in each region) belonging to each of 9 gradient orientation bins
    '''
    nrow = 5
    ncol = 5
    bin_n = 8
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray_old = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)
    flow = np.zeros((img_gray.shape[0], img_gray.shape[1], 2), np.float32)
    cv2.calcOpticalFlowFarneback(img_gray_old, img_gray, flow, 0.5, 1, 12, 2, 5, 1.1, 0)

    cells = np.asarray([np.hsplit(row, nrow) for row in np.vsplit(flow, ncol)])
    hists = np.asarray([])
    for row in range(nrow):
        for col in range(ncol):
            cellimg = cells[row, col]
            mag, ang = cv2.cartToPolar(cellimg[:,:,0], cellimg[:,:,1], angleInDegrees=True)
            '''
            We will use signed gradient
            '''
            # quantizing binvalues in (0...7)
            ang = np.mod(ang, 360)
            bins = np.int32(ang * bin_n / 360)
            hist = np.bincount(bins.ravel(), minlength=bin_n)
            '''
            Concatenate all of them, obtaining a total of 225 bins (i.e., an array with 225 values)
            '''
            hists = np.append(hists, hist)
    return hists

def draw_hof(oldframe, frame, filename):
    hof = getHOFperFrame(oldframe, frame)
    rows = frame.shape[0]
    cols = frame.shape[1]
    nrow = 5
    ncol = 5
    bin_n = 8
    length = 24.0
    img = np.zeros((rows, cols, 1), np.uint8)
    for row in range(nrow):
        for col in range(ncol):
            yc = int(rows / nrow * (row + 0.5))
            xc = int(cols / ncol * (col + 0.5))
            startpos = bin_n * (row * ncol + col)
            hist = hof[startpos:startpos+bin_n]
            for i in range(bin_n):
                angle = float(i * 360 / bin_n)
                dx, dy = cv2.polarToCart(np.array([length]), np.array([angle]), angleInDegrees=True)
                b = hist[i] / 3456 * 255
                cv2.line(img, (xc, yc), (xc+dx[0], yc+dy[0]), (b, b, b))
    cv2.imwrite(filename, img)

def classify_HOG_average(directory):
    hogvector_list = None
    train_list = None
    test_list = None
    labels = np.asarray([])
    train_labels = np.asarray([])
    test_labels = np.asarray([])

    '''
    2-3. For each video, compute the average of all per-frame HOG vectors in the video.
    Create a 2-D array with size 225*<num_videos>, by stacking averaged HOG vectors
    of all videos.
    '''
    for file in sorted(os.listdir(directory)):
        if file.endswith('.avi'):
            '''
            1. Load each video as a 3-D array (XYT data)
            '''
            print file
            label, data = xyt(directory + '/' + file)
            '''
            compute the average of all per-frame HOG vectors in the video
            '''
            framecount = data.shape[0]
            hists = None
            for frame in data:
                hog = getHOGperFrame(frame)
                if hists is None:
                    hists = hog
                else:
                    hists += hog
            hists /= framecount
            '''
            Create a 2-D array with size 225*<num_videos>
            '''
            if hogvector_list is None:
                hogvector_list = hists
            else:
                hogvector_list = np.vstack([hogvector_list, hists])

            '''
            Also maintain the activity IDs of the videos, by generating an array labels
            '''
            labels = np.append(labels, label)

            if int(file.split('_')[0]) <= 6:
                if train_list is None:
                    train_list = hists
                else:
                    train_list = np.vstack([train_list, hists])
                train_labels = np.append(train_labels, label)
            else:
                if test_list is None:
                    test_list = hists
                else:
                    test_list = np.vstack([test_list, hists])
                test_labels = np.append(test_labels, label)

    '''
    Try to visualize the resulting HOG vectors of 1_1 and 2_2. Try to create
    a grayscale image similar to HOG images of slide 21: draw one line per bin
    (I.e., 9 lines per spatial bin). The intensity of the line will be proportional
    to the value of the bin: the value 0 means intensity of 0, and the value
    3456 means intensity 255). Save the results as 1_1_hog.jpg and 2_2_hog.jpg
    '''
    label, data = xyt(directory + '/' + '1_1.avi')
    draw_hog(data[0], 'output/1_1_hog.jpg')
    label, data = xyt(directory + '/' + '2_2.avi')
    draw_hog(data[0], 'output/2_2_hog.jpg')

    '''
    2-4. Using the videos belonging to sets 1-6 (i.e., training videos), build the k-nearest neighbor (k-NN)
    classifier (k=3). Do the classification with the testing videos using the k-nearest neighbor classifier.
    Compare the classification result with the ground truth (i.e. labels). Measure classification accuracy:
    <num_correct>/<num_total_testing_vieo>
    '''
    num_correct = 0
    num_total_testing_video = test_list.shape[0]
    for testidx in range(test_list.shape[0]):
        dist = []
        test_hog = test_list[testidx]
        '''
        Use scipy.spatial.distance.euclidean to compute Euclidean distance.
        '''
        for train_hog in train_list:
            dist.append(euclidean(test_hog, train_hog))
        sortidx = np.asarray(dist).argsort()
        correct = False
        for i in range(3):
            if train_labels[sortidx[i]] == test_labels[testidx]:
                correct = True
                break
        if correct == True:
            num_correct += 1
    print '3-NN classification accuracy of HOG average: {:6.2f}%'.format(float(num_correct) / num_total_testing_video * 100)

def classify_HOF_average(directory):
    hofvector_list = None
    train_list = None
    test_list = None
    labels = np.asarray([])
    train_labels = np.asarray([])
    test_labels = np.asarray([])
    for file in sorted(os.listdir(directory)):
        if file.endswith('.avi'):
            '''
            1. Load each video as a 3-D array (XYT data)
            '''
            print file
            label, data = xyt(directory + '/' + file)
            '''
            compute the average of all per-frame hof vectors in the video
            '''
            framecount = data.shape[0]
            hists = None
            prev_img = None
            for frame in data:
                if not (prev_img is None):
                    hof = getHOFperFrame(prev_img, frame)
                    if hists is None:
                        hists = hof
                    else:
                        hists += hof
                prev_img = frame
            hists /= (framecount - 1)
            '''
            Create a 2-D array with size 225*<num_videos>
            '''
            if hofvector_list is None:
                hofvector_list = hists
            else:
                hofvector_list = np.vstack([hofvector_list, hists])

            '''
            Also maintain the activity IDs of the videos, by generating an array labels
            '''
            labels = np.append(labels, label)

            if int(file.split('_')[0]) <= 6:
                if train_list is None:
                    train_list = hists
                else:
                    train_list = np.vstack([train_list, hists])
                train_labels = np.append(train_labels, label)
            else:
                if test_list is None:
                    test_list = hists
                else:
                    test_list = np.vstack([test_list, hists])
                test_labels = np.append(test_labels, label)
    '''
    Try to visualize the resulting HOF vectors of 1_1 and 2_2. Try to create a grayscale HOF image
    similar to hof images (I.e., 8 lines per spatial bin). The intensity of the line will be
    proportional to the value of the bin: 0 means intensity 0, and 3456 means intensity 255).
    Save the results as 1_1_hof.jpg and 2_2_hof.jpg
    '''
    label, data = xyt(directory + '/' + '1_1.avi')
    draw_hof(data[0], data[1], 'output/1_1_hof.jpg')
    label, data = xyt(directory + '/' + '2_2.avi')
    draw_hof(data[0], data[1], 'output/2_2_hof.jpg')

    num_correct = 0
    num_total_testing_video = test_list.shape[0]
    for testidx in range(test_list.shape[0]):
        dist = []
        test_hof = test_list[testidx]
        '''
        Use scipy.spatial.distance.euclidean to compute Euclidean distance.
        '''
        for train_hof in train_list:
            dist.append(euclidean(test_hof, train_hof))
        sortidx = np.asarray(dist).argsort()
        correct = False
        for i in range(3):
            if train_labels[sortidx[i]] == test_labels[testidx]:
                correct = True
                break
        if correct == True:
            num_correct += 1
    print '3-NN classification accuracy of HOF average: {:6.2f}%'.format(float(num_correct) / num_total_testing_video * 100)

def classify_HOG_kmeans(directory):
    hogvector_list = None
    train_list = None
    test_list = None
    labels = np.asarray([])
    train_labels = np.asarray([])
    test_labels = np.asarray([])
    k = 400

    '''
    1. Sample 10 frames per video, and do k-means clustering (k=400) to find cluster centers based on
    those 10*<num_total_videos> samples.
    '''
    samplehog = None
    for file in sorted(os.listdir(directory)):
        if file.endswith('.avi'):
            '''
            1. Load each video as a 3-D array (XYT data)
            '''
            print file
            label, data = xyt(directory + '/' + file)
            '''
            compute the average of all per-frame HOG vectors in the video
            '''
            framecount = data.shape[0]
            samples = random.sample(range(framecount), 10)
            for sampleidx in samples:
                hog = getHOGperFrame(data[sampleidx])
                if samplehog is None:
                    samplehog = hog
                else:
                    samplehog = np.vstack([samplehog, hog])
    centroids, _ = kmeans(samplehog, k)

    for file in sorted(os.listdir(directory)):
        if file.endswith('.avi'):
            clusters = np.asarray([], np.int32)
            for frame in data:
                hog = getHOGperFrame(frame)
                idx, _ = vq([hog], centroids)
                clusters = np.append(clusters, idx)
            hist = np.bincount(clusters, minlength=k)
            norm = np.linalg.norm(hist)
            hists = hist / norm
            '''
            Create a 2-D array
            '''
            if hogvector_list is None:
                hogvector_list = hists
            else:
                hogvector_list = np.vstack([hogvector_list, hists])

            '''
            Also maintain the activity IDs of the videos, by generating an array labels
            '''
            labels = np.append(labels, label)

            if int(file.split('_')[0]) <= 6:
                if train_list is None:
                    train_list = hists
                else:
                    train_list = np.vstack([train_list, hists])
                train_labels = np.append(train_labels, label)
            else:
                if test_list is None:
                    test_list = hists
                else:
                    test_list = np.vstack([test_list, hists])
                test_labels = np.append(test_labels, label)

    '''
    2-4. Using the videos belonging to sets 1-6 (i.e., training videos), build the k-nearest neighbor (k-NN)
    classifier (k=3). Do the classification with the testing videos using the k-nearest neighbor classifier.
    Compare the classification result with the ground truth (i.e. labels). Measure classification accuracy:
    <num_correct>/<num_total_testing_vieo>
    '''
    num_correct = 0
    num_total_testing_video = test_list.shape[0]
    for testidx in range(test_list.shape[0]):
        dist = []
        test_hog = test_list[testidx]
        '''
        Use scipy.spatial.distance.euclidean to compute Euclidean distance.
        '''
        for train_hog in train_list:
            dist.append(euclidean(test_hog, train_hog))
        sortidx = np.asarray(dist).argsort()
        correct = False
        for i in range(3):
            if train_labels[sortidx[i]] == test_labels[testidx]:
                correct = True
                break
        if correct == True:
            num_correct += 1
    print '3-NN classification accuracy of HOG bag-of-words: {:6.2f}%'.format(float(num_correct) / num_total_testing_video * 100)


def classify_HOF_kmeans(directory):
    hofvector_list = None
    train_list = None
    test_list = None
    labels = np.asarray([])
    train_labels = np.asarray([])
    test_labels = np.asarray([])
    k = 400

    '''
    1. Sample 10 frames per video, and do k-means clustering (k=400) to find cluster centers based on
    those 10*<num_total_videos> samples.
    '''
    samplehof = None
    for file in sorted(os.listdir(directory)):
        if file.endswith('.avi'):
            '''
            1. Load each video as a 3-D array (XYT data)
            '''
            print file
            label, data = xyt(directory + '/' + file)
            '''
            compute the average of all per-frame HOF vectors in the video
            '''
            framecount = data.shape[0]
            samples = random.sample(range(framecount - 1), 10)
            for sampleidx in samples:
                hof = getHOFperFrame(data[sampleidx], data[sampleidx + 1])
                if samplehof is None:
                    samplehof = hof
                else:
                    samplehof = np.vstack([samplehof, hof])
    centroids, _ = kmeans(samplehof, k)

    for file in sorted(os.listdir(directory)):
        if file.endswith('.avi'):
            clusters = np.asarray([], np.int32)
            for frameidx in range(data.shape[0] - 1):
                hof = getHOFperFrame(data[frameidx], data[frameidx + 1])
                idx, _ = vq([hof], centroids)
                clusters = np.append(clusters, idx)
            hist = np.bincount(clusters, minlength=k)
            norm = np.linalg.norm(hist)
            hists = hist / norm
            '''
            Create a 2-D array
            '''
            if hofvector_list is None:
                hofvector_list = hists
            else:
                hofvector_list = np.vstack([hofvector_list, hists])

            '''
            Also maintain the activity IDs of the videos, by generating an array labels
            '''
            labels = np.append(labels, label)

            if int(file.split('_')[0]) <= 6:
                if train_list is None:
                    train_list = hists
                else:
                    train_list = np.vstack([train_list, hists])
                train_labels = np.append(train_labels, label)
            else:
                if test_list is None:
                    test_list = hists
                else:
                    test_list = np.vstack([test_list, hists])
                test_labels = np.append(test_labels, label)

    '''
    2-4. Using the videos belonging to sets 1-6 (i.e., training videos), build the k-nearest neighbor (k-NN)
    classifier (k=3). Do the classification with the testing videos using the k-nearest neighbor classifier.
    Compare the classification result with the ground truth (i.e. labels). Measure classification accuracy:
    <num_correct>/<num_total_testing_vieo>
    '''
    num_correct = 0
    num_total_testing_video = test_list.shape[0]
    for testidx in range(test_list.shape[0]):
        dist = []
        test_hof = test_list[testidx]
        '''
        Use scipy.spatial.distance.euclidean to compute Euclidean distance.
        '''
        for train_hof in train_list:
            dist.append(euclidean(test_hof, train_hof))
        sortidx = np.asarray(dist).argsort()
        correct = False
        for i in range(3):
            if train_labels[sortidx[i]] == test_labels[testidx]:
                correct = True
                break
        if correct == True:
            num_correct += 1
    print '3-NN classification accuracy of HOF bag-of-words: {:6.2f}%'.format(float(num_correct) / num_total_testing_video * 100)


def main():
    if not os.path.exists('output'):
        os.makedirs('output')

    '''
    Problem 1: Optical flows (10%)
    '''
    calcOpticalFlow('images/jpl_thomas', 'output/flow1.zip')
    calcOpticalFlow('images/dog', 'output/flow2.zip')

    '''
    Problem 2: Activity classification using HOG/HOF (60%)
    '''
    jpl_dir = 'images/jpl_interaction_segmented'

    classify_HOG_average(jpl_dir)

    '''
    5. Repeat the steps 1-4, while using histogram of 'optical flow orientations' instead of gradients
    '''
    classify_HOF_average(jpl_dir)

    '''
    Problem 3: Activity classification using HOG/HOF + bag-of-words (30%)
    '''
    '''
    1. For HOG, repeat the steps 1-4 of the above problem. However, this time, instead of averaging
    per-frame HOG descriptors, we will use bag-of-words representation to summarize them. [20%]
    '''
    classify_HOG_kmeans(jpl_dir)
    '''
    2. Do the above step 1 with HOF instead of HOG
    '''
    classify_HOF_kmeans(jpl_dir)
    
if __name__ == "__main__":
    main()
