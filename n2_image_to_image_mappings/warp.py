import homography

from PIL import Image
from matplotlib.tri import Triangulation
from numpy import *
from pylab import *
from scipy import ndimage


def image_in_image(im1, im2, tp):
    """ Put im1 in im2 with an affine transformation such that corners are as close to tp as possible.
    tp are homogeneous and counterclockwise from top left.
        e.g. [[tlx, blx, brx, trx],
              [tly, bly, bry, try],
              [  1,   1,   1,   1]]"""

    # points to warp from
    m,n = im1.shape[:2]
    fp = array([[0,m,m,0],[0,0,n,n],[1,1,1,1]])

    # compute affine transform and apply
    H = homography.Haffine_from_points(tp, fp)
    im1_t = ndimage.affine_transform(im1, H[:2,:2], (H[0,2],H[1,2]), im2.shape[:2])
    alpha = (im1_t > 0)

    return (1-alpha)*im2 + alpha*im1_t


def image_in_image_triangle(im1, im2, tp):
    """ Put im1 in im2 with an affine transformation such that corners are exactly at tp.
    tp are homogeneous and counterclockwise from top left. """

    # set from points to corners of im1
    m, n = im1.shape[:2]
    fp = array([[0, m, m, 0], [0, 0, n, n], [1, 1, 1, 1]])

    # first triangle
    tp2 = tp[:, :3]
    fp2 = fp[:, :3]
    # compute H
    H = homography.Haffine_from_points(tp2, fp2)
    im1_t = ndimage.affine_transform(im1, H[:2, :2], (H[0, 2], H[1, 2]), im2.shape[:2])

    # alpha for triangle. the reason why we need this, instead of alpha = (im1_t > 0):
    # we pass the whole im1 as the original image to affine_transform(), so that the result also contains the
    # whole transformed im1, instead of a triangle
    alpha = alpha_for_triangle(tp2, im2.shape[0], im2.shape[1])
    im3 = (1 - alpha) * im2 + alpha * im1_t

    # second triangle
    tp2 = tp[:, [0, 2, 3]]
    fp2 = fp[:, [0, 2, 3]]
    # compute H
    H = homography.Haffine_from_points(tp2, fp2)
    im1_t = ndimage.affine_transform(im1, H[:2, :2], (H[0, 2], H[1, 2]), im2.shape[:2])
    # alpha for triangle
    alpha = alpha_for_triangle(tp2, im2.shape[0], im2.shape[1])
    im4 = (1 - alpha) * im3 + alpha * im1_t

    return im4


def alpha_for_triangle(points,m,n):
    """ Creates alpha map of size (m,n) for a triangle with corners defined by points
    (given in normalized homogeneous coordinates). """

    alpha = zeros((m,n))
    for i in range(min(points[0]),max(points[0])):
        for j in range(min(points[1]),max(points[1])):
            x = linalg.solve(points,[i,j,1])
            if min(x) > 0:  # all coefficients positive
                alpha[i,j] = 1
    return alpha


def triangulate_points(x,y):
    """ triangulation of 2D points"""
    return Triangulation(x,y).triangles
    # each row contains the indices in the arrays x and y for the three points of each triangle


def pw_affine(fromim, toim, fp, tp, tri):
    """ Warp triangular patches from an image.
    fromim = image to warp
    toim = destination image
    fp = from points in hom. coordinates
    tp = to points in hom.  coordinates
    tri = triangulation. """

    im = toim.copy()

    # check if image is grayscale or color
    is_color = len(fromim.shape) == 3

    # create image to warp to (needed if iterate colors)
    im_t = zeros(im.shape, 'uint8')

    for t in tri:
        # compute affine transformation
        H = homography.Haffine_from_points(tp[:, t], fp[:, t])

        if is_color:
            for col in range(fromim.shape[2]):
                im_t[:,:,col] = ndimage.affine_transform(fromim[:,:,col],H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])
        else:
            im_t = ndimage.affine_transform(fromim,H[:2,:2],(H[0,2],H[1,2]),im.shape[:2])

        # alpha for triangle
        alpha = alpha_for_triangle(tp[:,t],im.shape[0],im.shape[1])

        # add triangle to image
        im[alpha>0] = im_t[alpha>0]

    return im


def plot_mesh(x,y,tri):
    """ Plot triangles. """
    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]] # add first point to end
        plot(x[t_ext],y[t_ext],'r')


def panorama(H,fromim,toim,padding=2400,delta=2400):
    """ Create horizontal panorama by blending two images using a homography H (preferably estimated using RANSAC).
    The result is an image with the same height as toim. 'padding'
    specifies number of fill pixels and 'delta' additional translation. """

    # check if images are grayscale or color
    is_color = len(fromim.shape) == 3

    # homography transformation for geometric_transform()
    def transf(p):      # not sure if this is right. might be p2=dot(linalg.inv(H),[p[0],p[1],1])
        p2 = dot(H,[p[0],p[1],1])
        return (p2[0]/p2[2],p2[1]/p2[2])

    if H[1,2]<0: # fromim is to the right   # not sure if it's correct. might be  if H[0,2]>0:
        print('warp - right')
        # transform fromim
        if is_color:
            # pad the destination image with zeros to the right
            toim_t = hstack((toim, zeros((toim.shape[0], padding, 3))))
            fromim_t = zeros((toim.shape[0], toim.shape[1] + padding, toim.shape[2]))

            for col in range(3):
                fromim_t[:,:,col] = ndimage.geometric_transform(fromim[:,:,col],
                       transf,(toim.shape[0],toim.shape[1]+padding))
        else:
            # pad the destination image with zeros to the right
            toim_t = hstack((toim, zeros((toim.shape[0], padding))))
            fromim_t = ndimage.geometric_transform(fromim,transf,
                      (toim.shape[0],toim.shape[1]+padding))
    else:
        print('warp - left')
        # add translation to compensate for padding to the left
        H_delta = array([[1,0,0],[0,1,-delta],[0,0,1]])     # not sure if it's correct
        H = dot(H,H_delta)

        # transform fromim
        if is_color:
            # pad the destination image with zeros to the left
            toim_t = hstack((zeros((toim.shape[0], padding, 3)), toim))
            fromim_t = zeros((toim.shape[0], toim.shape[1] + padding, toim.shape[2]))
            for col in range(3):
                fromim_t[:,:,col] = ndimage.geometric_transform(fromim[:,:,col],
                          transf,(toim.shape[0],toim.shape[1]+padding))
        else:
            # pad the destination image with zeros to the left
            toim_t = hstack((zeros((toim.shape[0], padding)), toim))
            fromim_t = ndimage.geometric_transform(fromim,
                      transf,(toim.shape[0],toim.shape[1]+padding))

    # blend and return (put fromim above toim)
    if is_color:
        # all non black pixels
        alpha = ((fromim_t[:,:,0] * fromim_t[:,:,1] * fromim_t[:,:,2] ) > 0)
        for col in range(3):
            toim_t[:,:,col] = fromim_t[:,:,col]*alpha + toim_t[:,:,col]*(1-alpha)
    else:
        alpha = (fromim_t > 0)
        toim_t = fromim_t*alpha + toim_t*(1-alpha)

    return toim_t


def example_rectangle():
    im1 = array(Image.open('continental.jpg').convert('L'))
    im2 = zeros([600,800], dtype = uint8)
    tp = array([[264,538,540,264],[40,36,605,605],[1,1,1,1]])

    im3 = image_in_image(im1, im2, tp)

    figure()
    gray()
    imshow(im3)
    axis('equal')
    axis('off')
    show()


def example_triangle():
    im1 = array(Image.open('continental.jpg').convert('L'))
    im2 = zeros([600,800], dtype = uint8)
    tp = array([[264, 538, 540, 264], [200, 36, 605, 400], [1, 1, 1, 1]])

    im3 = image_in_image_triangle(im1, im2, tp)

    figure()
    gray()
    imshow(im3)
    axis('equal')
    axis('off')
    show()


def example_panorama():
    import sift
    featname = ['part'+str(i)+'.sift' for i in range(2)]
    imname = ['part'+str(i)+'.jpg' for i in range(2)]
    l = {}
    d = {}
    for i in range(2):
        sift.process_image(imname[i], featname[i])
        l[i],d[i] = sift.read_features_from_file(featname[i])

    matches = {}
    for i in range(1):
        matches[i] = sift.match(d[i + 1], d[i])

    # function to convert the matches to hom. points
    def convert_points(j):
        ndx = matches[j].nonzero()[0]
        fp = homography.make_homog(l[j + 1][ndx, :2].T)
        ndx2 = [int(matches[j][i]) for i in ndx]
        tp = homography.make_homog(l[j][ndx2, :2].T)
        return fp, tp

    # estimate the homographies
    model = homography.RansacModel()

    fp, tp = convert_points(0)
    H_01 = homography.H_from_ransac(fp, tp, model)[0]  # im 0 to 1
    # fp, tp = convert_points(1)
    # H_12 = homography.H_from_ransac(fp, tp, model)[0]  # im 1 to 2
    #
    # fp, tp = convert_points(0)
    # H_01 = homography.H_from_ransac(fp, tp, model)[0]  # im 0 to 1
    #
    # tp, fp = convert_points(2)  # NB: reverse order
    # H_32 = homography.H_from_ransac(fp, tp, model)[0]  # im 3 to 2
    #
    # tp, fp = convert_points(3)  # NB: reverse order
    # H_43 = homography.H_from_ransac(fp, tp, model)[0]  # im 4 to 3

    # warp the images
    delta = 400  # for padding and translation

    im1 = array(Image.open(imname[0]))
    im2 = array(Image.open(imname[1]))
    im_12 = panorama(H_01, im1, im2, delta, delta)     # the order of im1 and im2 might be wrong

    # im1 = array(Image.open(imname[1]))
    # im2 = array(Image.open(imname[2]))
    # im_12 = panorama(H_12, im1, im2, delta, delta)
    #
    # im1 = array(Image.open(imname[0]))
    # im_02 = panorama(dot(H_12, H_01), im1, im_12, delta, delta)
    #
    # im1 = array(Image.open(imname[3]))
    # im_32 = panorama(H_32, im1, im_02, delta, delta)
    #
    # im1 = array(Image.open(imname[4]))
    # im_42 = panorama(dot(H_32, H_43), im1, im_32, delta, 2 * delta)

    Image.fromarray(uint8(im_12)).save('part_12.jpg')
    figure()
    imshow(uint8(im_12))
    axis('equal')
    axis('off')
    show()


# example_rectangle()
# example_triangle()
# example_panorama()
