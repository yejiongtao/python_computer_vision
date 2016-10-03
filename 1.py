from PIL import Image
from numpy.ma import sqrt
from pylab import *
import imtools
from scipy.ndimage import filters
import pydot


g = pydot.Dot(graph_type='graph')
g.add_node(pydot.Node(str(0)))
g.write_png('1.png')

