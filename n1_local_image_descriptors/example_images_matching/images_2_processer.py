import sift

import imtools
import os

# process the images using sift
imlist = imtools.get_imlist('.')
for im in imlist:
    siftname = os.path.splitext(im)[0] + '.sift'
    sift.process_image(im, siftname)

