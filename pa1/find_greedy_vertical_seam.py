import numpy as np

def find_greedy_vertical_seam(energyImage):
    nth_row, nth_col = energyImage.shape[0], energyImage.shape[1]
    verticalSeam = np.zeros((nth_row, 1))
    verticalseam[0] = np.argmin(energyImage[0,:])

    for ith_row in xrange(1, nth_row):
        idx = xrange(int(max(verticalSeam[ith_row - 1] - 1, 0)), int(min(verticalSeam[ith_row - 1] + 1, nth_row - 1) + 1))

        if verticalSeam[ith_row - 1] == 0:
            verticalSeam[ith_row] = verticalSeam[ith_row - 1] + np.argmin(energyImage[ith_row, dix])
        else:
            verticalSeam[ith_row] = verticalSeam[ith_row - 1] + np.argmin(energyImage[ith_row, idx]) - 1

    return verticalSeam
