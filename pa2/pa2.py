import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import os

def create_filter_bank(save_filter_image=False):
    '''
    1. Prepare filter banks for texture representation
    '''
    # get FIR of gaussian filters
    firimage = np.zeros((45, 45), np.float64)
    firimage[22, 22] = 1

    G2 = ndimage.filters.gaussian_filter(firimage, sigma=2)
    G4 = ndimage.filters.gaussian_filter(firimage, sigma=4)
    G8 = ndimage.filters.gaussian_filter(firimage, sigma=8)

    # generate filter bank
    filter_bank = (
        ndimage.convolve1d(G2, [1, -1], 1),
        ndimage.convolve1d(G2, [1, -1], 0),
        ndimage.convolve1d(G4, [1, -1], 1),
        ndimage.convolve1d(G4, [1, -1], 0),
        ndimage.convolve1d(G8, [1, -1], 1),
        ndimage.convolve1d(G8, [1, -1], 0),
        G8 - G2,
        G4 - G2
    )

    if save_filter_image:
        # save generated filters
        for i in range(len(filter_bank)):
            imgname = 'filter' + str(i+1) + '.jpg'
            plt.imsave('output/' + imgname, filter_bank[i], cmap=plt.cm.gray, format='jpg')

    return filter_bank


def convolve_filter(image, filter_bank, save_image = False, prefix = ''):
    '''
    2. Perform convolution using the created 8 different filters with the test images
    Measure each pixel's squared filter response per filter
    '''
    filter_response = []

    for i in range(len(filter_bank)):
        # convolution
        filtered = ndimage.convolve(image, filter_bank[i])
        filteredsq = np.square(filtered)
        # Rescale the array values so that its max value becomes 255
        maxv = np.max(np.max(filteredsq))
        filteredsq = filteredsq / maxv * 255
        filter_response.append(filteredsq)
        if (save_image):
            # save the result
            imgname = prefix + str(i+1) + '.jpg'
            plt.imsave('output/' + imgname, filteredsq, format='jpg')
    return filter_response

def texture_comparison(filter_response, save_texture = False, filename=''):
    h, w = filter_response[0].shape
    cx = int(w / 2)
    cy = int(h / 2)
    d = np.zeros((h, w), np.float64)
    # iterate through 8-D vector of each pixel in the image and compute Euclidean distance
    vc = np.asarray([filter_response[i][cy, cx] for i in range(8)])
    for y in range(h):
        for x in range(w):
            vp = np.asarray([filter_response[i][y, x] for i in range(8)])
            d[y, x] = np.linalg.norm(vc - vp)

    # save the result array D as a grayscale image by rescaling it
    maxv = np.max(np.max(d))
    d = d / maxv * 255
    if save_texture:
        plt.imsave('output/' + filename + '.jpg', d, cmap=plt.cm.gray, format='jpg')
    return d

def main():
    '''
    main funtion
    '''
    '''
    0. load zebra.jpg
    '''
    zebra_image = misc.imread('images/zebra.jpg', mode='F')
    if not os.path.exists('output'):
        os.makedirs('output')

    '''
    1. Prepare filter banks for texture representation
    '''
    filter_bank = create_filter_bank(True)

    '''
    2. Perform convolution using the created 8 different filters with the test images
    Measure each pixel's squared filter response per filter
    '''
    filter_response = convolve_filter(zebra_image, filter_bank, True, 'zebra_activation')
    '''
    3. Treat each pixel of the zebra image as a 8-D data, based on its squared filter responses
    Get the 8-D vector of the center pixel of the image. Next, iterate through 8-D vector of each pixel
    in the image, and compare it with the 8-D vector of the center pixel. Compute their Euclidean
    distance, and save them in a new array:
    '''
    texture_comparison(filter_response, True, filename='zebra_texture_comparison')

if __name__ == "__main__":
    main()
