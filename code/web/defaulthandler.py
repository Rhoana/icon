import tornado.ioloop
import tornado.web
import socket
import os;
import re
import StringIO
import glob

class DefaultHandler(tornado.web.RequestHandler):

        #def initialize(self, logic):
        #    print ('-->DefaultHandler.initialize...')
        #    self.__logic = logic
        def get(self):
            print ('-->DefaultHandler.get...')
            self.render("index.html")

        def post(self, uri):
            print ('-->DefaultHandler.post...')

        def close(self, signal, frame):
            print ('Saving..')
            sys.exit(0)


class Default(object):

  def __init__(self):
    self.__web_dir = 'web'

  def content_type(self, extension):
    return {
      '.js': 'text/javascript',
      '.html': 'text/html',
      '.json': 'application/javascript',
      '.png': 'image/png',
      '.jpg': 'image/jpg',
      '.ico': 'image/ico',
      '.ttf': 'font/ttf',
      '.woff': 'font/woff',
      '.woff2': 'font/woff2',
      '.map': 'text/html',
      '.css': 'text/css',
      '.cur': 'image/x-win-bitmap'
    }[extension]

  def handle(self, request):

    url = request.uri
    extension = os.path.splitext(url)[1]
    if self.content_type(extension) is None:
      return None, None

    # get filename from query
    requested_file = self.__web_dir + url

    print '-->requested_file:',requested_file

    if not os.path.exists(requested_file):
      print '-->eeee...requested_file:',requested_file
      return None, None

    with open(requested_file, 'r') as f:
      content = f.read()

    return content, self.content_type(extension)
