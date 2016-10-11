import json
from urllib import request

from pylab import *
from scipy import ndimage,linalg
from PIL import Image

import imtools
from mpl_toolkits.mplot3d import axes3d
from n1_local_image_descriptors import sift



b = 'nihao'.encode()
print(b.decode())