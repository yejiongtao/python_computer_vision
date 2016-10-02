from PIL import Image
from numpy.ma import sqrt
from pylab import *
import imtools
from scipy.ndimage import filters

pil_im = Image.open('continental.jpg').convert('L')
im = array(pil_im)

gray()

show()
