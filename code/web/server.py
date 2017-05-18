import tornado.ioloop
import tornado.web
import socket
import os
import sys
import time
import signal
# import datetime

import h5py


from datetime import datetime, date


import tornado.httpserver
from browserhandler import BrowseHandler
from annotationhandler import AnnotationHandler
from projecthandler import ProjectHandler
from helphandler import HelpHandler
from defaulthandler import DefaultHandler

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))

from utility import Utility
from database import Database
from paths import Paths

MAX_WAIT_SECONDS_BEFORE_SHUTDOWN = 0.5

class Application(tornado.web.Application):
    def __init__(self):
		handlers = [
			(r"/", DefaultHandler),
			(r"/browse.*", BrowseHandler),
			(r"/project.*", ProjectHandler),
            (r"/annotate.*", AnnotationHandler),
			(r'/help*', HelpHandler),
			(r'/settings/(.*)', tornado.web.StaticFileHandler, {'path': 'resources/settings/'}),
			(r'/js/(.*)', tornado.web.StaticFileHandler, {'path': 'resources/js/'}),
			(r'/js/vendors/(.*)', tornado.web.StaticFileHandler, {'path': 'resources/js/vendors/'}),
			(r'/css/(.*)', tornado.web.StaticFileHandler, {'path': 'resources/css/'}),
			(r'/uikit/(.*)', tornado.web.StaticFileHandler, {'path': 'resources/uikit/'}),
			(r'/images/(.*)', tornado.web.StaticFileHandler, {'path': 'resources/images/'}),
			(r'/open-iconic/(.*)', tornado.web.StaticFileHandler, {'path': 'resources/open-iconic/'}),
            (r'/input/(.*)', tornado.web.StaticFileHandler, {'path': 'resources/input/'}),
            (r'/train/(.*)', tornado.web.StaticFileHandler, {'path': 'resources/input/'}),
            (r'/validate/(.*)', tornado.web.StaticFileHandler, {'path': 'resources/input/'}),
			#(r"/annotate/(.*)", AnnotationHandler, dict(logic=self)),
		]

		settings = {
			"template_path": 'resources',
			"static_path": 'resources',
		}

		tornado.web.Application.__init__(self, handlers, **settings)


import numpy as np
class Server():
    def __init__(self, name, port):
        self.name = name
        self.port = port
        application = Application()
        self.http_server = tornado.httpserver.HTTPServer( application )
        hostname = socket.gethostname()
        print 'hostname:', hostname
        self.ip = hostname #socket.gethostbyname( hostname )

    def print_status(self):
        Utility.print_msg ('.')
        Utility.print_msg ('\033[93m'+ self.name + ' running/' + '\033[0m', True)
        Utility.print_msg ('.')
        Utility.print_msg ('open ' + '\033[92m'+'http://' + self.ip + ':' + str(self.port) + '/' + '\033[0m', True)
        Utility.print_msg ('.')

    def start(self):
        self.print_status()
        self.http_server.listen( self.port )
        tornado.ioloop.IOLoop.instance().start()

    def stop(self):
        msg = 'shutting down %s in %s seconds'%(self.name, MAX_WAIT_SECONDS_BEFORE_SHUTDOWN)
        Utility.print_msg ('\033[93m'+ msg + '\033[0m', True)
        io_loop = tornado.ioloop.IOLoop.instance()
        deadline = time.time() + MAX_WAIT_SECONDS_BEFORE_SHUTDOWN

        def stop_loop():
            now = time.time()
            if now < deadline and (io_loop._callbacks or io_loop._timeouts):
                io_loop.add_timeout(now + 1, stop_loop)
            else:
                io_loop.stop()
                Utility.print_msg ('\033[93m'+ 'shutdown' + '\033[0m', True, 'done')
        stop_loop()

def sig_handler(sig, frame):
    msg = 'caught interrupt signal: %s'%sig
    Utility.print_msg ('\033[93m'+ msg + '\033[0m', True)
    tornado.ioloop.IOLoop.instance().add_callback(shutdown)

def shutdown():
    server.stop()

def main():
    global server
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)

    port = 8888
    name = 'icon webserver'
    server = Server(name, port)
    server.start()

if __name__ == "__main__":
    main()
