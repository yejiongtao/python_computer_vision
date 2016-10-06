from numpy.core.umath import sign

from numpy.ma import exp
import numpy
from pylab import *
from scipy import linalg
from PIL import Image


class Camera(object):
    """ Class for representing pin-hole cameras. """

    def __init__(self, P):
        """ Initialize P = K[R|t] camera model. """
        self.P = P
        self.K = None  # calibration matrix
        self.R = None  # rotation
        self.t = None  # translation
        self.c = None  # camera center

    def project(self, X):
        """  Project points in X (4*n array) and normalize coordinates. """
        x = dot(self.P, X)
        for i in range(3):
            x[i] /= x[2]
        return x

    def factor(self):
        """  Factorize the camera matrix into K,R,t as P = K[R|t]. """
        # factor first 3*3 part
        K, R = linalg.rq(self.P[:, :3])

        # make diagonal of K positive
        T = diag(sign(diag(K)))
        if linalg.det(T) < 0:
            T[1, 1] *= -1

        self.K = dot(K, T)
        self.R = dot(T, R)  # T is its own inverse
        self.t = dot(linalg.inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t

    def center(self):
        """  Compute and return the camera center. """
        if self.c is not None:
            return self.c
        else:
            # compute c by factoring
            self.factor()
            self.c = -dot(self.R.T, self.t)
            return self.c


def rotation_matrix(a):
    """  Creates a 3D rotation matrix for rotation around the axis of the vector a. """
    R = eye(4)
    R[:3,:3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return R


def my_calibration(sz):
    """ calculates K based on the given image size.
    sz: size of the image """
    row, col = sz
    fx = 2555 * col / 2592      # 2555 is the fx calculated using an image with width 2592
    fy = 2586 * row / 1936      # 2586 is the fy calculated using an image with height 1936
    K = diag([fx, fy, 1])
    K[0, 2] = 0.5 * col         # assumming the optical axis intersects with the image at the center of the image
    K[1, 2] = 0.5 * row
    return K


def example_project_3d():
    # load points
    points = loadtxt('house.p3d').T
    points = vstack((points, ones(points.shape[1])))

    # setup camera
    P = hstack((eye(3), array([[0], [0], [-10]])))
    cam = Camera(P)
    x = cam.project(points)

    # plot projection
    figure()
    plot(x[0], x[1], 'k.')

    # create transformation
    r = 0.05 * numpy.random.random(3)
    rot = rotation_matrix(r)

    # rotate camera and project
    figure()
    for t in range(20):
        cam.P = dot(cam.P, rot)
        x = cam.project(points)
        plot(x[0], x[1], 'k.')
    show()


def cube_points(c,wid):
    """  Creates a list of points for plotting
      a cube with plot. (the first 5 points are
      the bottom square, some sides repeated). """
    p = []
    # bottom
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])  # same as first to close plot

    # top
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])  # same as first to close plot

    # vertical sides
    p.append([c[0] - wid, c[1] - wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] + wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] + wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])

    return array(p).T


def example_pose_estimation():
    from n2_image_to_image_mappings import homography
    from n1_local_image_descriptors import sift

    # compute features
    sift.process_image('book_frontal.JPG', 'im0.sift')
    l0, d0 = sift.read_features_from_file('im0.sift')

    sift.process_image('book_perspective.JPG', 'im1.sift')
    l1, d1 = sift.read_features_from_file('im1.sift')

    # match features and estimate homography
    matches = sift.match_twosided(d0, d1)
    ndx = matches.nonzero()[0]
    fp = homography.make_homog(l0[ndx, :2].T)
    ndx2 = [int(matches[i]) for i in ndx]
    tp = homography.make_homog(l1[ndx2, :2].T)

    model = homography.RansacModel()
    H = homography.H_from_ransac(fp, tp, model)[0]

    # camera calibration
    K = my_calibration((300, 400))

    # 3D points at plane z=0 with sides of length 0.2
    box = cube_points([0, 0, 0.1], 0.1)

    # project bottom square in first image
    cam1 = Camera(hstack((K, dot(K, array([[0], [0], [-1]])))))
    # first points are the bottom square
    box_cam1 = cam1.project(homography.make_homog(box[:, :5]))

    # compute second camera matrix from cam1 and H
    cam2 = Camera(dot(H, cam1.P))
    A = dot(linalg.inv(K), cam2.P[:, :3])
    A = array([A[:, 0], A[:, 1], cross(A[:, 0], A[:, 1])]).T
    cam2.P[:, :3] = dot(K, A)

    # project with the second camera
    box_cam2 = cam2.project(homography.make_homog(box))

    im0 = array(Image.open('book_frontal.JPG'))
    im1 = array(Image.open('book_perspective.JPG'))

    # 2D projection of bottom square
    figure()
    imshow(im0)
    plot(box_cam1[0, :], box_cam1[1, :], linewidth=3)

    # 3D cube
    figure()
    imshow(im1)
    plot(box_cam2[0, :], box_cam2[1, :], linewidth=3)

    show()

    # import pickle
    #
    # with open('ar_camera.pkl', 'bw') as f:
    #     pickle.dump(K, f)
    #     pickle.dump(dot(linalg.inv(K), cam2.P), f)    # Rt


# example_project_3d()
# example_pose_estimation()

