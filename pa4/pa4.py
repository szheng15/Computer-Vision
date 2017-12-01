from scipy import ndimage
from PIL import Image
from array import array
from skimage import feature, io

import numpy as np
import matplotlib.pyplot as plt

def loadImage(file_name):
    im = Image.open(file_name).convert("L")
    data = np.array(im)
    return data

im = loadImage('line_original.jpg')
print im

def detectEdges(img, sig):
    output = np.uint8(feature.canny(img, sigma = sig) * 255)

    return output


edges1 = detectEdges(im, 1)
edges2 = detectEdges(im, 3)


#Q2&3
def houghTransform(im):
    theta = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = im.shape
    rmax = np.ceil(np.sqrt(width**2 + height**2))
    rhos = np.linspace(-rmax, rmax, rmax * 2.0)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    num_theta = len(theta)
    H = np.zeros((2 * rmax, num_theta), dtype = np.uint64)
    y_idx, x_idx = np.nonzero(im)
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        for j in range(0, num_theta, 3):
            rho = round(x * cos_t[j] + y * sin_t[j]) + rmax
            H[rho, j/3] += 1
    return H, theta, rhos

def prepareImage(arr):
    arr_max = float(np.amax(arr))
    arr_min = float(np.amin(arr))
    return np.uint8(((arr - arr_min) / (arr_max - arr_min)) * 255)

H, theta, rhos = houghTransform(edges2)

i = Image.fromarray(prepareImage(H))
i.save("line_hough.jpg", "JPEG")
i.show()


#Q4

def prepareCoords(coords, H):
    new_c = []
    for c in coords:
        rho = rhos[c/ H.shape[1]]
        theta_new = theta[c % H.shape[1]]
    return new_c

coordinates = prepareCoords(peak_local_max(H, min_distance = 20), H)
print coordinates

fig, ax = plt.subplots(1,2, figsize = (10, 10))

ax[0].imshow(im, cmap = plt.cm.gray)
ax[0].set_title('Input Image')
ax[0].axis(' image')

ax[1].imshow(H, cmap = 'jet', extent = [np.deg2rad(theta[-1]), np.deg2rad(theta[0]), rhos[-1], rhos[0]])

ax[1].set_aspect('equal', adjustable = 'box')
ax[1].set_title('Hough Transform')
ax[1].axis('image')

plt.show()

# fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey= True, subplot_kw={'adjustable': 'box-forced'})

# ax = axes.ravel()
# ax[0].imshow(im, cmap=plt.cm.gray)
# ax[0].axis('off')
# ax[0].set_title('Original')

# ax[1].imshow(im, cmap=plt.cm.gray)
# ax[1].autoscale(False)
# ax[1].plot(280, 0, 'r')
# ax[1].axis('off')
# ax[1].set_title('Peak Local Max')

# ax[2].imshow(im, cmap=plt.cm.gray)
# ax[2].axis('off')
# ax[2].set_title('Original')

# plt.show()


# Q5

imx = np.zeros(im.shape)
filters.sobel(edges2, 1, imx)
imy = np.zeros(im.shape)

filters.sobel(edges2, 0, imy)

magnitude = np.sqrt(imx**2 + imy**2)

thet = np.arctan(imy/imx)

pil_im = Image.fromarray(np.uint8(thet))
pil_im.show()

def houghTransform(im, direction):
    theta = np.deg2rad(np.arange(-90.0 , 90.0))
    width,height = im.shape
    rmax = np.ceil(np.sqrt(width * width + height * height))
    rhos = np.linspace(-rmax, rmax, rmax * 2.0)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    num_theta = len(theta)
    H = np.zeros((2 * rmax, num_theta), dtype = np.uint64)
    y_idx, x_idx = np.nonzero(im)
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        theta_new = direction[x_idx][y_idx]
        for j in range(0, num_theta, 3):
            rho = round(x * cos_t[j] + y * sin_t[j]) + rmax
            H[rho, j/3] += 1
    return H, theta, rhos
