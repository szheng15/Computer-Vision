from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

def desplaySeam(im, seam, type):
    plt.figure(1)
    if type == 'HORIZONTAL':
        plt.plot(range(len(seam)), seam)
        plt.title('Horizontal Seam')

    elif type == 'VERTICAL':
        plt.plot(seam, range(len(seam)))
        plt.title('Vertical Seam')

    else:
        print "Error in displaySeam"


    plt.hold(True)
    plt.imshow(im)
    plt.show()
#imsave('seam_v1.jpg')
