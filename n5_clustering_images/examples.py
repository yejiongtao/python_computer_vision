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

    figure()
    imshow(codeim)
    show()


example_K_means_fonts()
# example_K_means_pixel()
