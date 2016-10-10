import imtools

from PIL import Image, ImageDraw
from numpy import *
from numpy.ma import floor
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


def example_project_on_first_2_pc():
    """ visualize projecting on the first 2 principal components.
     projecting on 2 pc results in a rank 2 vector. """

    # get list of images
    imlist = imtools.get_imlist('fonts/')
    imnbr = len(imlist)

    # load model file
    with open('font_pca_modes.pkl','rb') as f:
        immean = pickle.load(f)
        V = pickle.load(f)

    # create matrix to store all flattened images
    immatrix = array([array(Image.open(im)).flatten() for im in imlist],'f')

    # project on the first and second components
    projected = array([dot(V[[0, 2]], immatrix[i] - immean) for i in range(imnbr)])
    # height and width
    h, w = 1200, 1200

    # create a new image with a white background
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # draw axis
    draw.line((0, h / 2, w, h / 2), fill=(255, 0, 0))
    draw.line((w / 2, 0, w / 2, h), fill=(255, 0, 0))

    # scale coordinates to fit
    scale = abs(projected).max(0)
    scaled = floor(array([(p / scale) * (w / 2 - 20, h / 2 - 20) +
                          (w / 2, h / 2) for p in projected]))

    # paste thumbnail of each image
    for i in range(imnbr):
        nodeim = Image.open(imlist[i])
        nodeim.thumbnail((25, 25))
        ns = nodeim.size
        img.paste(nodeim, (int(scaled[i][0] - ns[0] // 2), int(scaled[i][1] -
                           ns[1] // 2), int(scaled[i][0] + ns[0] // 2 + 1), int(scaled[i][1] + ns[1] // 2 + 1)))
    figure()
    imshow(img)
    show()


# example_pca()
example_project_on_first_2_pc()