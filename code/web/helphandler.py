import tornado.ioloop
import tornado.web
import socket
import os
import re
import glob

class HelpHandler(tornado.web.RequestHandler):

    def get(self):
        print ('-->HelpHandler.get...' + self.request.uri)
        self.render("help.html")
