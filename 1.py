from PIL import Image
from numpy.ma import sqrt
from pylab import *
import imtools
from scipy.ndimage import filters

pil_im = Image.open('continental.jpg')
pil_im.resize((1000, 600)).save('continental_resized.jpg')