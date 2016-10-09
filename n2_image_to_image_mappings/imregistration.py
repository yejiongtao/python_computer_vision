import imtools
from n1_local_image_descriptors import sift
from n2_image_to_image_mappings import homography

import os
from numpy import *
from PIL import Image
from pylab import *
from scipy import ndimage


def compute_rigid_transform(refpoints,points):
    """ Computes rotation, scale and translation for aligning points to refpoints. """

    A = array([[points[0][0], -points[0][1], 1, 0],
               [points[0][1], points[0][0], 0, 1],
               [points[1][0], -points[1][1], 1, 0],
               [points[1][1], points[1][0], 0, 1],
               [points[2][0], -points[2][1], 1, 0],
               [points[2][1], points[2][0], 0, 1]])

    y = array([refpoints[0][0],
               refpoints[0][1],
               refpoints[1][0],
               refpoints[1][1],
               refpoints[2][0],
               refpoints[2][1]])

    # least sq solution to mimimize ||Ax - y||
    a, b, tx, ty = linalg.lstsq(A, y)[0]
    R = array([[a, -b], [b, a]])  # rotation matrix incl scale

    return R, tx, ty


def rigid_alignment(imlist, points):
    """  Align images rigidly and plot the images. """

    imlen = len(imlist)

    # take the points in the first image as reference points
    refpoint = points[0]
    refim = array(Image.open(imlist[0]))

    figure()
    gray()
    axis('off')
    axis('equal')

    subplot(2, imlen, 1)
    imshow(refim)
    subplot(2, imlen, 1+imlen)
    imshow(refim)

    # warp each image using affine transform
    for i in range(1, imlen):
        point = points[i]

        R, tx, ty = compute_rigid_transform(refpoint, point)
        T = array([[R[1][1], R[1][0]], [R[0][1], R[0][0]]])

        im = array(Image.open(imlist[i]))
        im2 = ndimage.affine_transform(im, linalg.inv(T), offset=[-ty, -tx])

        subplot(2, imlen, i+1)
        imshow(im)
        subplot(2, imlen, i+1+imlen)
        imshow(im2)


def annotate_images():
    for no in range(1, 11):
        no = str(no)
        imshow(array(Image.open('att_faces/s6/'+no+'.pgm')))
        gray()
        a = ginput(3)
        savetxt('annotations/s6/' + no + '.txt',a)


# annotate_images()

# the images are grayscale and of the same size
impath = 'att_faces/s6'
imlist = [os.path.join(impath, f) for f in os.listdir(impath) if f.endswith('.pgm')]

points_path = 'annotations/s6'
points = []
for im in imlist:
    imname = os.path.splitext(os.path.basename(im))[0]
    points.append(loadtxt(os.path.join(points_path, imname +'.txt')))
    # there are three points in a file, namely the right corner of the left eye, the left corner of the right eye,
    # and the middle of the mouth

rigid_alignment(imlist, points)

show()

