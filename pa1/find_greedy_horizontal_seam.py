import numpy as numpy

def find_greedy_horizontal_seam(energyImage):
    nth_row, nth_col = energyImage.shape[0], energyImage.shape[1]
    horizontalSeam = np.zeros((nth_col, 1))
    horizontalSeam[0] = np.argmin(energyImage[:,0])

    for ith_col in xrange(1, nth_col):
        idx = xrange(int(max(horizontalSeam[ith_col - 1] - 1, 0)), int(min(horizontalSeam[ith_col - 1] + 1, nth_col - 1) + 1))

        if horizontalSeam[ith_col - 1] == 0:
            horizontalSeam[ith_col] = horizontalSeam[ith_col - 1] + np.argmin(energyImage[idx, ith_col])
        else:
            horizontalSeam[ith_col] = horizontalSeam[ith_col - 1] + np.argmin(energyImage[idx, ith_col]) - 1

    return horizontalSeam
