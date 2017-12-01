import numpy as np
from scipy.ndimage.filters import convolve
import skimage
from skimage import color

def energy_image(im):
    input_color_image = skimage.img_as_float(color.rgb2gray(im))
    gradient_x = convolve(input_color_image, np.array([[1,-1]]), mode = "wrap")
    gradient_y = convolve(input_color_image, np.array([[1],[-1]]), mode = "wrap")
    energyImage = np.sqrt((gradient_x**2)+(gradient_y**2))
    return energyImage


test1 = array(Image.open('seam_carving_input1.jpg'))
energyImage = energy_image(test1)
#imsave('energy_image1.jpg', energyImage)
#plot.imshow(energyImage)
img = Image.fromarray(energyImage)
img.show()

#keep return me the black image, can't fiugre out where is the problem



