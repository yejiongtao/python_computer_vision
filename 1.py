from pylab import *
from scipy import ndimage,linalg
from PIL import Image

import imtools
from mpl_toolkits.mplot3d import axes3d

imlist = imtools.get_imlist('n0_general/fonts')
for im in imlist:
    Image.open(im).convert('L').save(im)