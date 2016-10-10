from urllib import request

url = 'https://www.flickr.com/search/?text=aurora%20polaris'
c = request.urlopen(url)
http = c.read().decode('utf-8')

imurls = []
start = 0
while True:
    index = http.find('img.src', start)
    if index == -1:
        break

    imurls.append('http:' + http[index+9 : http.find(';', index)-1])
    start = index+1

for i,url in enumerate(imurls):
    image = request.URLopener()
    image.retrieve(url, 'images/' + str(i) + '.jpg')
    print('downloading',url)