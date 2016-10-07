from n3_camera_models_and_augmented_reality import camera
from n4_multiple_view_geometry import sfm
from PIL import Image
from pylab import *
from numpy import *


def example_plot_3d_points():
    # plotting 3D points
    from mpl_toolkits.mplot3d import axes3d
    fig = figure()
    ax = fig.gca(projection='3d')
    ax.plot(points3D[0],points3D[1],points3D[2],'k.')


def example_compute__and_plot_epipole():
    # index for points in first two views
    ndx = (corr[:, 0] >= 0) & (corr[:, 1] >= 0)

    # get coordinates and make homogeneous
    x1 = points2D[0][:, corr[ndx, 0]]
    x1 = vstack((x1, ones(x1.shape[1])))
    x2 = points2D[1][:, corr[ndx, 1]]
    x2 = vstack((x2, ones(x2.shape[1])))

    # compute F
    F = sfm.compute_fundamental(x1, x2)

    # compute the epipole
    e = sfm.compute_epipole(F)

    # plotting
    figure()
    imshow(im1)
    # plot each line individually, this gives nice colors
    for i in range(5):
        sfm.plot_epipolar_line(im1, F, x2[:, i], e, False)
    axis('off')

    figure()
    imshow(im2)
    # plot each point individually, this gives same colors as the lines
    for i in range(5):
        plot(x2[0, i], x2[1, i], 'o')
    axis('off')


def example_triangulation():
    # index for points in first two views
    ndx = (corr[:, 0] >= 0) & (corr[:, 1] >= 0)

    # get coordinates and make homogeneous
    x1 = points2D[0][:, corr[ndx, 0]]
    x1 = vstack((x1, ones(x1.shape[1])))
    x2 = points2D[1][:, corr[ndx, 1]]
    x2 = vstack((x2, ones(x2.shape[1])))

    Xtrue = points3D[:, ndx]
    Xtrue = vstack((Xtrue, ones(Xtrue.shape[1])))

    # check first 3 points
    Xest = sfm.triangulate(x1, x2, P[0].P, P[1].P)
    print(Xest[:, :3])
    print(Xtrue[:, :3])

    # plotting
    from mpl_toolkits.mplot3d import axes3d
    fig = figure()
    ax = fig.gca(projection='3d')
    ax.plot(Xest[0], Xest[1], Xest[2], 'ko')
    ax.plot(Xtrue[0], Xtrue[1], Xtrue[2], 'r.')
    axis('equal')


def example_compute_P_from_points():
    """ pick out the points that are visible in the first view, compute the camera matrix from them.
    and use the estimated P to project the 3d points to see the outcome """
    corr1 = corr[:, 0]  # view 1
    ndx3D = where(corr1 >= 0)[0]  # missing values are -1
    ndx2D = corr1[ndx3D]

    # select visible points and make homogeneous
    x = points2D[0][:, ndx2D]  # view 1
    x = vstack((x, ones(x.shape[1])))
    X = points3D[:, ndx3D]
    X = vstack((X, ones(X.shape[1])))

    # estimate P
    Pest = camera.Camera(sfm.compute_P(x, X))

    # compare!
    print(Pest.P / Pest.P[2, 3])
    print(P[0].P / P[0].P[2, 3])

    xest = Pest.project(X)

    # plotting
    figure()
    imshow(im1)
    plot(x[0], x[1], 'bo')
    plot(xest[0], xest[1], 'r.')
    axis('off')


# load some images
im1 = array(Image.open('images/001.jpg'))
im2 = array(Image.open('images/002.jpg'))

# load 2D points for each view to a list
points2D = [loadtxt('2D/00'+str(i+1)+'.corners').T for i in range(3)]

# load 3D points
points3D = loadtxt('3D/p3d').T

# load correspondences
corr = genfromtxt('2D/nview-corners',dtype='int',missing_values='*')

# load cameras to a list of Camera objects
P = [camera.Camera(loadtxt('2D/00'+str(i+1)+'.P')) for i in range(3)]

# example_plot_3d_points()
# example_compute__and_plot_epipole()
# example_triangulation()
example_compute_P_from_points()
show()




