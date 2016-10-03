import json
import os
from urllib import request


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
    image = request.URLopener()
    image.retrieve(url, os.path.basename(request.urlparse(url).path))
    print('downloading:', url)

