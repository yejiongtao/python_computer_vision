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


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):   # to use this, denoise(im, im), do not understand..
    """ An implementation of the Rudin-Osher-Fatemi (ROF) denoising model
    using the numerical procedure presented in eq (11) A. Chambolle (2005).

    Input: noisy input image (grayscale), initial guess for U, weight of
    the TV-regularizing term, steplength, tolerance for stop criterion.

    Output: denoised and detextured image, texture residual. """

    m,n = im.shape      # size of noisy image

    # initialize
    U = U_init
    Px = im             # x-component to the dual field
    Py = im             # y-component of the dual field
    error = 1

    while error > tolerance:
        Uold = U

        # gradient of primal variable
        GradUx = roll(U, -1, axis=1)-U      # x-component of U's gradient
        GradUy = roll(U, -1, axis=0)-U      # y-component of U's gradient

        # update the dual varible
        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = maximum(1,sqrt(PxNew**2+PyNew**2))

        Px = PxNew/NormNew                  # update of x-component (dual)
        Py = PyNew/NormNew                  # update of y-component (dual)

        # update the primal variable
        RxPx = roll(Px,1,axis=1)            # right x-translation of x-component
        RyPy = roll(Py,1,axis=0)            # right y-translation of y-component

        DivP = (Px-RxPx)+(Py-RyPy)          # divergence of the dual field.
        U = im + tv_weight*DivP             # update of the primal variable

        # update of error
        error = linalg.norm(U-Uold)/sqrt(n*m)

    return U,im-U                           # denoised image and texture residual


