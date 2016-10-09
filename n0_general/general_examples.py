import imtools

from PIL import Image
from numpy import *
from pylab import *
import os
import pickle


def example_pca(save=False):
    imlist = imtools.get_imlist('fonts')

    im = array(Image.open(imlist[0])) # open one image to get size
    m,n = im.shape[0:2] # get the size of the images

    # create matrix to store all flattened images
    immatrix = array([array(Image.open(im).convert('L')).flatten() for im in imlist],'f')

    # perform PCA
    V,S,immean = imtools.pca(immatrix)

    if save:
        with open('font_pca_modes.pkl', 'wb') as f:
            pickle.dump(immean, f)
            pickle.dump(V, f)

    # show some images (mean and 7 first modes)
    figure()
    gray()
    subplot(2,4,1)
    imshow(immean.reshape(m,n))
    for i in range(7):
        subplot(2,4,i+2)
        imshow(V[i].reshape(m,n))

    show()

example_pca()