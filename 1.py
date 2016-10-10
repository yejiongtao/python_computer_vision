import json
from urllib import request

from pylab import *
from scipy import ndimage,linalg
from PIL import Image

import imtools
from mpl_toolkits.mplot3d import axes3d


url = 'http://www.panoramio.com/map/get_panoramas.php?order=popularity&' + \
    'set=public&from=0&to=20&minx=-77.037564&miny=38.896662&maxx=-77.035564&maxy=38.898662&size=medium'
c = request.urlopen(url)

str = c.read().decode('utf-8')
print(str)

j = json.loads(str)
imurls = []
for im in j['photos']:
    imurls.append(im['photo_file_url'])

for url in imurls:
    print('downloading:', url)

