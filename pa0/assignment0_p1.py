import numpy as np
from PIL import Image



def convolve(image, filter):
    y = np.zeros((image.shape[0] - filter.shape[0] + 1, image.shape[1] - filter.shape[1] + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            for a in range(filter.shape[0]):
                for b in range(filter.shape[1]):
                    y[i,j] += filter[a,b] * image[i+a,j+b]
    return y


img = np.array(Image.open('empire.jpg')) * 1.0

Gaussian_filter = np.array([[1,2,1],[2,4,2],[1,2,1]]) * (1/16)
#convolve(img, Gaussian_filter)

# I think my function is good, but I can't figure out why it can not apply to the image.






