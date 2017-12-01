from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_vertical_map import find_optimal_vertical_map
import numpy as np
from energy_image import energy_image

def reduceWidth(im, energyImage):
    input_color_image = im

    input_image_nth_row, input_image_nth_col, input_image_nth_channel = input_color_image.shape

    reducedColorImage = np.zeros((input_image_nth_row, input_image_nth_col - 1,input_image_nth_channel), ntype = np.uint8)

    verticalSeam = find_optimal_vertical_map(cumulative_minimum_energy_map(energyImage, 'VERTICAL'))

    for ith_row in xrange(0, input_image_nth_row):
        reducedColorImage[ith_row, :, 0] = np.delete(input_color_image[ith_row, :, 0], verticalSeam[ith_row])
        reducedColorImage[ith_row, :, 1] = np.delete(input_color_image[ith_row, :, 1], verticalSeam[ith_row])
        reducedColorImage[ith_row, :, 2] = np.delete(input_color_image[ith_row, :, 2], verticalSeam[ith_row])

    reducedEnergyImage = energy_image(reducedColorImage)

    return reducedColorImage, reducedEnergyImage
