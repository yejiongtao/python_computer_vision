import os
from PIL import Image
from numpy.ma import sqrt, maximum
from pylab import *


def get_imlist(path):
    """ returns a list of filenames for all jpg images in a folder"""
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def convert_to_jpg(filelist):
    """ converts a list of image files into jpg format"""
    for infile in filelist:
        outfile = os.path.splitext(infile)[0] + '.jpg'
        if infile != outfile:
            try:
                Image.open(infile).save(outfile)
            except IOError:
                print("cannot convert", infile)


def imresize(im, sz):
    """ resize an image array using PIL """
    pil_im = Image.fromarray(uint8(im))
    return array(pil_im.resize(sz))


def histeq(im, nbr_bins = 256):
    """ histogram equalization of a grayscale image """
    imhist, bins = histogram(im.flatten(), nbr_bins, normed = True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]

    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf


def compute_average(imlist):
    """ comput the average of a list of images """
    averageim = array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += array(Image.open(imname))
        except:
            print(imname, "skipped")
    averageim /= len(imlist)
    return array(averageim, 'uint8')


def pca(X):
    """ Principle Component Analysis
        input: X, matrix with training data stored as flattened arrays in rows
        return: projection matrix (with important dimensions first), variance and mean. """
    num_data, dim = X.shape

    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:          # PCA, compact trick used
        M = dot(X, X.T)         # covariance matrix
        e, EV = linalg.eigh(M)  # eigenvalues and eigenvectors
        tmp = dot(X.T, EV).T    # this is the compact trick
        V = tmp[::-1]           # reverse since last eigenvalues are the ones we want
        S = sqrt(e)[::-1]       # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:                       # PCA, SVD used
        U,S,V = linalg.svd(X)
        V = V[:num_data]        # only makes sense to return the first num_data

    return V,S,mean_X           # return the projection matrix, the variance and the mean



