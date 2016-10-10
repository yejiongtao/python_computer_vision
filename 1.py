import json
from urllib import request

from pylab import *
from scipy import ndimage,linalg
from PIL import Image

import imtools
from mpl_toolkits.mplot3d import axes3d
from n1_local_image_descriptors import sift


imlist = imtools.get_imlist('n6_searching_images/images')
nbr_images = len(imlist)
featlist = [ imlist[i][:-3]+'sift' for i in range(nbr_images)]

for i in range(nbr_images):
    sift.process_image(imlist[i],featlist[i])

