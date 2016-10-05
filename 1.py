from PIL import Image
from numpy.ma import sqrt
from pylab import *
import numpy
import imtools
from scipy.ndimage import filters
import pydot
from matplotlib.tri import Triangulation
import sift
from scipy import ndimage

a = arange(12.).reshape((4, 3))
def shift_func(output_coords):
    return (output_coords[0] - 0.5, output_coords[1] - 0.5)
b = ndimage.geometric_transform(a, shift_func)
print(b)