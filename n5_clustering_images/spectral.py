import pickle

import imtools
from PIL import Image
from numpy.ma import sqrt, minimum
from scipy.cluster.vq import *
from numpy import *
from pylab import *


def example_spectral_fonts():
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
    projected = array([dot(V[:20],immatrix[i]-immean) for i in range(imnbr)])

    n = len(projected)

    # compute distance matrix
    S = array([[sqrt(sum((projected[i] - projected[j]) ** 2))
                for i in range(n)] for j in range(n)], 'f')

    # create Laplacian matrix
    rowsum = sum(S, axis=0)
    D = diag(1 / sqrt(rowsum))
    I = identity(n)
    L = I - dot(D, dot(S, D))

    # compute eigenvectors of L
    U, sigma, V = linalg.svd(L)

    k = 5
    # create feature vector from k first eigenvectors
    # by stacking eigenvectors as columns
    features = array(V[:k]).T

    # k-means
    features = whiten(features)
    centroids, distortion = kmeans(features, k)
    code, distance = vq(features, centroids)

    # plot clusters
    for c in range(k):
        ind = where(code == c)[0]
        figure()
        for i in range(minimum(len(ind), 39)):
            im = Image.open(imlist[ind[i]])
            subplot(4, 10, i + 1)
            imshow(array(im))
            axis('equal')
            axis('off')
        gray()
    show()


def example_spectral_images():
    imlist = imtools.get_imlist('../n1_local_image_descriptors/example_images_matching')
    n = len(imlist)

    # load the similarity matrix and reformat
    S = loadtxt('../n1_local_image_descriptors/example_images_matching/matchscores.txt')
    S = 1 / (S + 1e-6)  # invert the scores so that the smaller, the more similar.
                        # a small number is added to avoid division with zero

    # create Laplacian matrix
    rowsum = sum(S, axis=0)
    D = diag(1 / sqrt(rowsum))
    I = identity(n)
    L = I - dot(D, dot(S, D))

    # compute eigenvectors of L
    U, sigma, V = linalg.svd(L)

    k = 2
    # create feature vector from k first eigenvectors
    # by stacking eigenvectors as columns
    features = array(V[:k]).T

    # k-means
    features = whiten(features)
    centroids, distortion = kmeans(features, k)
    code, distance = vq(features, centroids)

    # plot clusters
    for c in range(k):
        ind = where(code == c)[0]
        figure()
        for i in range(minimum(len(ind), 39)):
            im = Image.open(imlist[ind[i]])
            subplot(4, 4, i + 1)
            imshow(array(im))
            axis('equal')
            axis('off')
        gray()
    show()


# example_spectral_fonts()
example_spectral_images()
