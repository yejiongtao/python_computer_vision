import os
from pylab import *
import sift
import pydot
from PIL import Image


path = './'
imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
featlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sift')]

nbr_images = len(imlist)

matchscores = zeros((nbr_images,nbr_images))
for i in range(nbr_images):
    for j in range(i,nbr_images): # only compute upper triangle
        print('comparing ', imlist[i], imlist[j])

        l1,d1 = sift.read_features_from_file(featlist[i])
        l2,d2 = sift.read_features_from_file(featlist[j])

        matches = sift.match_twosided(d1,d2)

        nbr_matches = sum(matches > 0)
        print('number of matches = ', nbr_matches)
        matchscores[i,j] = nbr_matches

savetxt('matchscores.txt', matchscores)
