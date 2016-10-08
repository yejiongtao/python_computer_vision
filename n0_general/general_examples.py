import imtools

from PIL import Image
from numpy import *
from pylab import *
import os


def example_pca():
    path = 'att_faces/s2'
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pgm')]

    im = array(Image.open(imlist[0])) # open one image to get size
    m,n = im.shape[0:2] # get the size of the images
    imnbr = len(imlist) # get the number of images

    # create matrix to store all flattened images
    immatrix = array([array(Image.open(im)).flatten() for im in imlist],'f')

    # perform PCA
    V,S,immean = imtools.pca(immatrix)

    # show some images (mean and 7 first modes)
    figure()
    gray()
    subplot(2,4,1)
    imshow(immean.reshape(m,n))
    for i in range(7):
        subplot(2,4,i+2)
        imshow(V[i].reshape(m,n))

    show()