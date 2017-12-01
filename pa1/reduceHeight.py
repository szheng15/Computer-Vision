from cumulative_minimum_energy_map import cumulative_minimum_energy_map
from find_optimal_horizontal_seam import find_optimal_horizontal_seam
import numpy as np
from energy_image import energy_image

def reduceHeight(im, energyImage):
    input_color_image = im
    input_image_nth_row, input_image_nth_col, input_image_nth_channel = input_color_image.shape

    reducedColorImage = np.zeros((input_image_nth_row - 1, input_image_nth_col, input_image_nth_channel), dtype = np.uint8)
    horizontalSeam = find_optimal_horizontal_seam(cumulative_minimum_energy_map(energyImage, 'HORIZONTAL'))

    for ith_col in xrange(0, input_image_nth_col):
        reducedColorImage[:, ith_col, 0] = np.delete(input_color_image[:, ith_col,0],horizontalSeam[ith_col])
        reducedColorImage[:, ith_col, 1] = np.delete(input_color_image[:, ith_col,1],horizontalSeam[ith_col])
        reducedColorImage[:, ith_col, 2] = np.delete(input_color_image[:, ith_col,2],horizontalSeam[ith_col])

    reducedEnergyImage = energy_image(reducedColorImage)

    return reducedColorImage, reducedEnergyImage
