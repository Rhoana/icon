import tornado.ioloop
import tornado.web
import socket
import shutil
import os
import sys
import re
import glob
import json
base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))
sys.path.insert(2,os.path.join(base_path, '../database'))

from db import DB
from project import Project
from utility import Utility
from settings import Paths
#from performance import Performance
from db import Label
from db import Image

class ProjectHandler(tornado.web.RequestHandler):

    def get(self):
        print ('-->ProjectHandler.get...' + self.request.uri)
        tokens = self.request.uri.split(".")
        print tokens

        if len(tokens) > 1 and tokens[1] == 'getprojects':
            #self.render( self.getimages() )
            print 'ProjectHandler.getting projects...'
            self.set_header('Content-Type', 'text')
            projects = json.dumps(DB.getProjects())
            self.write(Utility.compress( projects ))
        elif len(tokens) > 2 and tokens[1] == 'getproject':
            #self.render( self.getimages() )
            print 'ProjectHandler.getting project...'
            self.set_header('Content-Type', 'text')
            projects = json.dumps(DB.getProject( tokens[2] ).toJson() )
            self.write(Utility.compress( projects ))
        elif len(tokens) > 2 and tokens[1] == 'getprojecteditdata':
            #self.render( self.getimages() )
            print 'ProjectHandler.getprojecteditdata...'
            self.set_header('Content-Type', 'text')
            self.write(self.getProjectEditData( tokens[2] ))
        else:
            self.render("project.html")

    def post(self):
        print ('-->ProjectHandler.post...')
        print ('-->ProjectHandler.post...' + self.request.uri)
        tokens  = self.request.uri.split(".")
        if len(tokens) > 2 and tokens[1] == 'removeproject':
            projectId = tokens[2]
            self.remove_project( projectId )
        elif len(tokens) > 1 and tokens[1] == 'saveproject':
            data = self.get_argument("data", default=None, strip=False)
            project =  json.loads( data )
            self.save_project( project )


    def getProjectEditData(self, projectId ):
        print 'project.getProjectEditData ', projectId
        project = DB.getProject( projectId )
        project = None if project is None else project.toJson()
        data = {}
        data['project'] = project
        data['images'] = Utility.getImages( Paths.TrainGrayscale )
        data['validation_images'] = Utility.getImages( Paths.ValidGrayscale )
        data['projectnames'] = DB.getProjectNames()
        data = json.dumps( data )
        return Utility.compress( data )

    def remove_project(self, projectId):
        print 'removing project....', projectId
        images = DB.getImages( projectId )
        for image in images:
            DB.removeImage(projectId, image.id)

        DB.removeProject( projectId )

        # remove learning model
        types = ['mlp', 'cnn']
        for t in types:
            #best_cnn_model.cnn.0.pklA
            path = '%s/best_%s.%s.*.pkl'%(Paths.Models,projectId,t)
            paths = glob.glob(path)
            for path in paths:
                print 'deleting model...', path
                os.remove( path )

        # remove labels
        path = '%s/*.%s.json'%(Paths.Labels, projectId)
        labels = glob.glob(path)
        for p in labels:
            print 'trying to delete:', p
            os.remove( p )

        # remove segmentations
        path = '%s/*.%s.seg'%(Paths.Segmentation, projectId)
        segs = glob.glob(path)
        for p in segs:
            print 'trying to delete:', p
            os.remove( p )


    def setupModel( self, project ):
        projectId = project['id']
        baseProjectId = project['initial_model']
        modeltype = project['model_type']

        modeltype = modeltype.lower()
        srcpath = '%s/best_%s_model.%s.pkl'%(Paths.Models, modeltype, baseProjectId)
        if not os.path.exists( srcpath ):
            return

        dstpath = '%s/best_%s_model.%s.pkl'%(Paths.Models, modeltype, projectId)
        if os.path.exists( dstpath ):
            return

        shutil.copyfile( srcpath, dstpath )

    def copyAnnotations( self, image, project ):
        imageId = image['image_id']
        projectId = project['id']
        baseProjectId = project['initial_model']

        srcFile = '%s/%s.%s.json'%(Paths.Labels, imageId, baseProjectId)
        dstFile = '%s/%s.%s.json'%(Paths.Labels, imageId, projectId)

        if not os.path.exists( srcFile ):
            return

        if os.path.exists( dstFile ):
            os.remove( dstFile )

        shutil.copyfile( srcFile, dstFile )


    def store_images(self, project, images, purpose=0):
        # store images added by the user
        image_names = DB.getImageNames( project['id'], purpose )
        for image in images:
            #self.copy_annotations( image, project )
            image = Image( id=image['image_id'], purpose=purpose)
            DB.storeImage(project['id'], project['model_type'], image )
            if image.id in image_names:
                image_names.remove(image.id)

        # discard images removed by the user
        for image_name in image_names:
            DB.removeImage(project['id'], image_name )

    def save_project(self, project):

        p_existing = DB.getProject( project['id'] )
        image_names = DB.getImageNames( project['id'] )

        jsonProject = Project.fromJson( project )

        if p_existing is None:
            print '------'
            print project

            # add the project to the database
            DB.storeProject( jsonProject )

            self.setupModel( project )

            # create the baseline performance metrics
            #Performance.measureBaseline( project['id'] )
        else:
            DB.updateProject( jsonProject )

        # create hidden layers
        #for units in project['hidden_layers']:
        #    DB.storeHiddenUnits(project['id'], project['model_type'], units)
        #DB.storeHiddenUnits(project['id'], project['model_type'], json.loads( project['hidden_layers'] ))
        print 'hidden_layers:'
        print type(project['hidden_layers']), project['hidden_layers']
        DB.storeHiddenUnits(project['id'], project['model_type'], project['hidden_layers'] )

        # copy annotations
        for image in project['images']:
            self.copyAnnotations( image, project )

        # store labels
        for label in project['labels']:
            DB.storeLabel(
            project['id'],
            project['model_type'],
            Label(
            label['index'],
            label['name'],
            label['r'],label['g'],label['b']) )

        self.store_images(project, project['images'], purpose=0)
        self.store_images(project, project['validation_images'], purpose=1)

        # # store images added by the user
        # image_names = DB.getImageNames( project['id'] )
        # for image in project['images']:
        #     #self.copy_annotations( image, project )
        #     image = Image( image['image_id'] )
        #     DB.storeImage(project['id'], project['model_type'], image )
        #     if image.id in image_names:
        #     image_names.remove(image.id)
        #
        # # discard images removed by the user
        # for image_name in image_names:
        #     DB.removeImage(project['id'], image_name )
