import cherrypy, os, urllib, pickle
from n6_searching_images import imagesearch
from n6_searching_images.vocabulary import Vocabulary

import random


class SearchDemo(object):
    def __init__(self):
        # load list of images
        with open('ukbench_imlist.pkl', 'rb') as f:
            self.imlist = pickle.load(f)

        self.nbr_images = len(self.imlist)
        self.ndx = list(range(self.nbr_images))

        # load vocabulary
        with open('vocabulary.pkl', 'rb') as f:
            self.voc = pickle.load(f)

        # set max number of results to show
        self.maxres = 15

        # header and footer html
        self.header = """
          <!doctype html>
          <head>
          <title>Image search example</title>
          </head>
          <body>
          """
        self.footer = """
          </body>
          </html>
          """

    def index(self, query=None):
        self.src = imagesearch.Searcher('test.db', self.voc)

        html = self.header
        html += """
          <br />
          Click an image to search. <a href='?query='>Random selection</a> of images.
          <br /><br />
          """
        if query:
            # query the database and get top images
            res = self.src.query(query)[:self.maxres]
            for dist, ndx in res:
                imname = self.src.get_filename(ndx)
                html += "<a href='?query=" + imname + "'>"
                html += "<img src='" + imname + "' width='100' />"
                html += "</a>"
        else:
            # show random selection if no query
            random.shuffle(self.ndx)
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                html += "<a href='?query=" + imname + "'>"
                html += "<img src='" + imname + "' width='100' />"
                html += "</a>"

        html += self.footer
        return html

    index.exposed = True


config = {
    'global':{
        'server.socket_host': "127.0.0.1",
        'server.socket_port': 8080,
        'server.thread_pool': 50,
        'tools.sessions.on': True
    },
    '/': {
        'tools.staticdir.root': os.path.abspath(os.path.dirname(__file__)),
        'tools.staticdir.on': True,
        'tools.staticdir.dir': ''
    }
}
cherrypy.quickstart(SearchDemo(), '/', config=config)