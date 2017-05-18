#-------------------------------------------------------------------------------------------
# browse.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains database access layer implementation footer
#           sqlite3
#-------------------------------------------------------------------------------------------

import tornado.ioloop
import tornado.web
import socket
import os
import sys
import re
import glob
import json
import StringIO
import numpy as np

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))
sys.path.insert(2,os.path.join(base_path, '../database'))

from utility import Utility
from paths import Paths
from db import DB
from h5data import H5Data
from PIL import Image

DATA_PATH_IMAGES = os.path.join(base_path, '../../data/input')
DATA_PATH = os.path.join(base_path, '../../data')
DATA_NAME = 'main'


class BrowseHandler(tornado.web.RequestHandler):


    # def __init__(self):
    #     self.requestcount = 0;

    def get(self):
        #self.requestcount += 1

        print ('-->BrowseHandler.get...' + self.request.uri)
        print (self.request.uri )
        #print ('requestcount: ' + self.requestcount )
        tokens = self.request.uri.split(".")
        print tokens

        if len(tokens) > 2 and tokens[1] == 'getimages':
            print 'getting images...'
            self.set_header('Content-Type', 'text')
            self.write(self.getimages( tokens[2] ))

        elif len(tokens) > 2 and tokens[1] == 'getpreview':
            self.set_header('Content-Type', 'image/tiff')
            self.write(self.getimage( tokens[2] ))


        elif len(tokens) > 2 and tokens[1] == 'getprojectsdata':
            print 'getprojectsdata...'
            self.set_header('Content-Type', 'text')
            self.write(self.getProjectsData( tokens[2] ))

        elif len(tokens) > 1 and tokens[1] == 'getallimages':
            self.set_header('Content-Type', 'text')
            self.write(self.getallimages())
        else:
            print 'returning default...'
            self.render("browser.html")

    def post(self):
        print ('-->BrowseHandler.post...', self.request.uri)
        tokens = self.request.uri.split(".")
        if len(tokens) > 2 and tokens[1] == 'stop':
            DB.stopProject( tokens[2] )
        elif len(tokens) > 2 and tokens[1] == 'start':
            print 'tokens:', tokens[2], tokens[1]
            DB.startProject( tokens[2] )


    def getimage(self, imageId):
        image = H5Data.get_slice(DATA_PATH, DATA_NAME, imageId )
        image = Image.fromarray(np.uint8(image*255))
        image.thumbnail((525,525), Image.ANTIALIAS)
        output = StringIO.StringIO()
        image.save(output, 'TIFF')
        return output.getvalue()


    def getProjectsData(self, projectId):
        print 'browse.getProjectEditData'

        project          = DB.getProject( projectId )
        data             = {}
        data['names']    = DB.getProjectNames()
        #data['num_slices']    = H5Data.getSliceCount()
        #data['num_slices'] = self.getSliceCount()

        data['num_slices'] = H5Data.getSliceCount(Paths.Data)

        if project == None and len(data['names']) > 0:
            project  = DB.getProject( data['names'][0] )

        active           = DB.getActiveProject()
        data['project']  = project.toJson()
        #DB.getProject( projectId ).toJson()
        #data['images']   = [ i.toJson() for i in DB.getImages( projectId ) ]
        #data['offline']  = DB.getOfflinePerformance( projectId )
        #data['online']   = DB.getOnlinePerformance( projectId )
        #data['baseline'] = DB.getBaselinePerformance( projectId )
        data['active']   = active.toJson() if active is not None else {}

        return Utility.compress( json.dumps( data ) )

    def close(self, signal, frame):
        print ('Saving..')
        sys.exit(0)


