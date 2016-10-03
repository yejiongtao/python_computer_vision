import pydot
import os
from PIL import Image
from pylab import *


path = '.'
fullpath = 'C:/Users/jioye/Desktop/work/codes/pycharm/CV/example_images_matching/'
imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
featlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sift')]
matchscores = loadtxt('matchscores.txt')

nbr_images = len(imlist)
threshold = 2       # min number of matches needed to create link

g = pydot.Dot(graph_type='graph')
for i in range(nbr_images):
    for j in range(i+1,nbr_images):
        if matchscores[i,j] > threshold:
            # first image in pair
            im = Image.open(imlist[i])
            im.thumbnail((100,100))
            filename = str(i)+'.png'
            im.save(filename)       # need temporary files of the right size
            g.add_node(pydot.Node(str(i),fontcolor='transparent',
                shape='rectangle',image=fullpath+filename))

            # second image in pair
            im = Image.open(imlist[j])
            im.thumbnail((100,100))
            filename = str(j)+'.png'
            im.save(filename)       # need temporary files of the right size
            g.add_node(pydot.Node(str(j),fontcolor='transparent',
                shape='rectangle',image=fullpath+filename))

            g.add_edge(pydot.Edge(str(i),str(j)))

g.write_png('whitehouse.png')

imshow(Image.open('whitehouse.png'))
axis('off')
show()