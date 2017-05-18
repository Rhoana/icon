import tornado.ioloop
import tornado.web
import socket
import time
import os
import sys
import zlib
import StringIO
import base64
import numpy as np;
import json
import h5py
from PIL import Image
from datetime import datetime, date
from scipy.misc import imread

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))
sys.path.insert(2,os.path.join(base_path, '../database'))

DATA_PATH_IMAGES = os.path.join(base_path, '../../data/input')
DATA_PATH_SEGMENTATION = os.path.join(base_path, '../../data/segmentation')
DATA_PATH = os.path.join(base_path, '../../data')
DATA_PATH_LABELS = os.path.join(base_path, '../../data/labels')
DATA_NAME = 'main'

from db import DB
from paths import Paths
from utility import Utility;
from h5data import H5Data


class AnnotationHandler(tornado.web.RequestHandler):

    DefaultProject = 'default'

    def get(self):
        print ('-->AnnotationHandler.get...' + self.request.uri)
        #self.__logic.handle( self );
        tokens  = self.request.uri.split(".")
        #typeId = tokens[1]
        imageId = tokens[1]
        projectId = tokens[2]
        action  = None 
        if (len(tokens) >= 4):
            action = tokens[3]

        purpose = tokens[1]
        if (purpose =='train' or purpose=='valid'):
            imageId = tokens[2]
            projectId = tokens[3]

        if action == 'getlabels':
            self.set_header('Content-Type', 'text')
            self.write(self.getLabels( imageId, projectId ))
        elif action == 'getimage':
            self.set_header('Content-Type', 'image/tiff')
            self.write(self.getimage( imageId, projectId ))
        elif action == 'setimagepurpose':
            self.setimagepurpose( imageId, projectId, purpose)
            # self.set_header('Content-Type', 'image/tiff')
            # self.write(self.setimagepurpose( imageId, projectId, purpose))

        # elif action == 'getpreviewimage':
        #     self.set_header('Content-Type', 'image/jpeg')
        #     self.write(self.getimage( imageId, projectId ))
        elif action == 'getuuid':
            #uuid.uuid1()
            guid = tokens[4]
            self.set_header('Content-Type', 'application/octstream')
            self.write(self.getuuid(projectId, imageId, guid))
        elif action == 'getannotations':
            self.set_header('Content-Type', 'text')
            self.write(self.getAnnotations( imageId, projectId ))
        elif action == 'getsegmentation':
            self.set_header('Content-Type', 'application/octstream')
            segTime = None if (len(tokens) < 5) else tokens[4]
            self.write(self.getsegmentation( imageId, projectId, segTime ))
        elif action == 'getstatus':
            guid = tokens[4]
            segTime = tokens[5]
            self.set_header('Content-Type', 'application/octstream')
            self.write(self.getstatus( imageId, projectId, guid, segTime ))
        else:
            self.render("annotate.html")

            

    def post(self):
        tokens  = self.request.uri.split(".")
        action=tokens[1]

        if action == 'saveannotations':
            data = self.get_argument("annotations", default=None, strip=False)
            imageId = self.get_argument("id", default=None, strip=False)
            projectId = self.get_argument("projectid", default=None, strip=False)
            self.saveannotations(imageId, projectId, data)
        elif action == 'setpurpose':
            purpose = self.get_argument("purpose", default=None, strip=False)
            imageId = self.get_argument("id", default=None, strip=False)
            projectId = self.get_argument("projectid", default=None, strip=False)
            DB.addImage(projectId, imageId, purpose)


    def getimage(self, imageId, projectId):
        image = H5Data.get_slice(DATA_PATH, DATA_NAME, imageId )
        image = Image.fromarray(np.uint8(image*255))
 
        output = StringIO.StringIO()
        image.save(output, 'TIFF')
        return output.getvalue()

    def renderimage(self, projectId, imageId, purpose):
        H5Data.extract_to(DATA_PATH, DATA_NAME, DATA_PATH_IMAGES, projectId, imageId, purpose )

        self.render("annotate.html")


    def getuuid(self, projectId, imageId, guid):
        data = {}
        project = DB.getProject( projectId )
        task = DB.getImage( projectId, imageId )

        expiration = project.syncTime*4

        if task.annotationLockId == guid:
            data['uuid'] = DB.lockImage( projectId, imageId )
            now = datetime.now()
            annotationTime = datetime.strptime(task.annotationTime, '%Y-%m-%d %H:%M:%S')
            diff = now - annotationTime
        elif task.annotationStatus == 1:
            now = datetime.now()
            annotationTime = datetime.strptime(task.annotationTime, '%Y-%m-%d %H:%M:%S')
            diff = now - annotationTime
            diff = diff.total_seconds()
            if diff > expiration:
                data['uuid'] = DB.lockImage( projectId, imageId )
        else:
            data['uuid'] = DB.lockImage( projectId, imageId )

        return Utility.compress(json.dumps( data ))

    def getLabels(self, imageId, projectId):
        path = 'resources/labels/%s.%s.json'%(imageId,projectId)
        content = '[]'
        try:
            with open(path, 'r') as content_file:
                content = content_file.read()
        except:
            pass
        return Utility.compress(content)

    def getAnnotations(self, imageId, projectId):

        path = 'resources/labels/%s.%s.json'%(imageId,projectId)
        # check the incoming folder first before to ensure
        # the most latest data is being referenced.
        path_incoming = 'resources/incoming/%s.%s.json'%(imageId,projectId)
        path = path_incoming if os.path.exists(path_incoming) else path

        #default to the labels template
        content = '[]'
        try:
            with open(path, 'r') as content_file:
                content = content_file.read()
        except:
            pass

        return Utility.compress(content)

    def saveannotations(self, imageId, projectId, data):
        # Always save the annotations to the labels folder.
        path = '%s/%s.%s.json'%(Paths.Labels, imageId,projectId)
        with open(path, 'w') as outfile:
            outfile.write(data)

        # Add a training and prediction task to the database
        DB.saveAnnotations( projectId, imageId, path )

        H5Data.generate_preview( DATA_PATH, DATA_NAME, DATA_PATH_LABELS, DATA_PATH_SEGMENTATION, DATA_PATH_IMAGES, imageId, projectId )

        images = DB.getTrainingImages( projectId )
        for img in images:
            print img.id, img.annotationFile, img.annotationTime, img.annotationStatus



    def has_new_segmentation(self, imageId, projectId, segTime):
        # if no new segmentation, just return nothing
        if segTime is None or segTime == 'undefined':
            return True

        task = DB.getImage(projectId, imageId)
        taskSegTime = time.strptime(task.segmentationTime, '%Y-%m-%d %H:%M:%S')
        segTime = segTime.replace("%20", " ")
        segTime = time.strptime(segTime, '%Y-%m-%d %H:%M:%S')

        if segTime == taskSegTime:
            return False

        return True


    def getsegmentation(self, imageId, projectId, segTime):
        data = []
        # if no new segmentation, just return nothing
        if not self.has_new_segmentation(imageId, projectId, segTime):
            return Utility.compress(data)

        path = 'resources/output/%s.%s.seg'%(imageId,projectId)
        data = []
        # Settings.addPredictionImage( projectId, imageId)
        if os.path.isfile( path ):
            with open(path, 'r') as content_file:
                compressed = content_file.read()
                decompressed = zlib.decompress(compressed)
                data = base64.b64decode(decompressed)
        return Utility.compress(data)

    def getstatus(self, imageId, projectId, guid, segTime):
        # make sure this image prioritize for segmentation
        DB.requestSegmentation( projectId, imageId )
        task = DB.getImage(projectId, imageId);
        data = {}
        data['image'] = task.toJson()
        data['project'] = DB.getProject(projectId).toJson()
        data['has_new_segmentation'] = self.has_new_segmentation(imageId, projectId, segTime)
        return Utility.compress(json.dumps( data ))
