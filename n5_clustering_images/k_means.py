import imtools
import pickle

from numpy.ma import minimum, floor
from scipy.cluster.vq import *
from scipy.misc import imresize
from numpy import *
from PIL import Image, ImageDraw
from pylab import *


def example_K_means_fonts():
    # get list of images
    imlist = imtools.get_imlist('../n0_general/fonts/')
    imnbr = len(imlist)

    # load model file
    with open('../n0_general/font_pca_modes.pkl','rb') as f:
        immean = pickle.load(f)
        V = pickle.load(f)

    imshape = array(Image.open(imlist[0])).shape
    # create matrix to store all flattened images
    immatrix = array([array(Image.open(im)).flatten() for im in imlist],'f')

    # project on the 20 first PCs
    immean = immean.flatten()
    projected = array([dot(V[:20],immatrix[i]-immean) for i in range(imnbr)])  # how to decide the 20????????????????????

    # k-means
    projected = whiten(projected)
    centroids,distortion = kmeans(projected,4)

    code,distance = vq(projected,centroids)

    # plot clusters
    for k in range(4):
        ind = where(code==k)[0]
        figure()
        gray()
        for i in range(minimum(len(ind),40)):
            subplot(4,10,i+1)
            imshow(immatrix[ind[i]].reshape(imshape))
            axis('off')
    show()


def example_K_means_pixel():
    steps = 100  # image is divided in steps*steps region
    im = array(Image.open('empire.jpg'))

    dx = im.shape[0] / steps
    dy = im.shape[1] / steps

    # compute color features for each region
    features = []
    for x in range(steps):
        for y in range(steps):
            R = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 0])
            G = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 1])
            B = mean(im[x * dx:(x + 1) * dx, y * dy:(y + 1) * dy, 2])
            features.append([R, G, B])
    features = array(features, 'f')  # make into array

    # cluster
    centroids, variance = kmeans(features, 3)
    code, distance = vq(features, centroids)

    # create image with cluster labels
    codeim = code.reshape(steps, steps)
    codeim = imresize(codeim, im.shape[:2], interp='nearest')
        # use 'nearest' to interpolate so that no new pixel values are introduced

    figure()
    imshow(codeim)
    show()


# example_K_means_fonts()
example_K_means_pixel()
