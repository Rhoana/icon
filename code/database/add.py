#---------------------------------------------------------------------------
# database.py
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
#---------------------------------------------------------------------------

import os;
import sqlite3 as lite
import sys
import json
import glob
import time
import uuid

from datetime import datetime, date

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))

from utility import *
from paths import Paths
from tables import Tables
from project import Project
from db import DB

DATABASE_NAME = os.path.join(base_path, '../../data/database/icon.db')

def install(project):
    # remove any existing model files associated with this project
    path = '%s/best_%s.%s.pkl'%(Paths.Models, project.id, project.type )
    path = path.lower()
    if os.path.exists( path ):
        os.remove( path )

    # remove any existing model files associated with this project
    path = '%s/best_%s.%s.offline.pkl'%(Paths.Models, project.id, project.type )
    path = path.lower()
    if os.path.exists( path ):
        os.remove( path )

    # install the default model
    project.addLabel( 0, 'background', 255,0,0)
    project.addLabel( 1, 'membrane', 0,255,0)

    # setup training set
    paths = glob.glob('%s/*.tif'%(Paths.TrainGrayscale))
    paths.sort()

    # setup the first 20 images as a training set
    i = 0
    for path in paths:
        name = Utility.get_filename_noext( path )
        segFile = '%s/%s.%s.seg'%(Paths.Segmentation, name, project.id)
        annFile = '%s/%s.%s.json'%(Paths.Labels, name, project.id)
        segFile = segFile.lower()
        annFile = annFile.lower()
        if not os.path.exists( segFile ):
            segFile = None
        if not os.path.exists( annFile ):
            annFile = None

        print 'adding image: %s ann: %s'%(name, annFile)
        project.addImage( imageId=name, annFile=annFile, segFile=segFile)
        i += 1
        if i > 20:
            break

    # store the project
    DB.storeProject( project )


#---------------------------------------------------------------------------
# Entry point to the main function of the program.
#---------------------------------------------------------------------------
if __name__ == '__main__':

    print 'adding project...'

    if True:
        print 'updating database...'
        projectId = 'janeliacnn'
        projectType = 'CNN'
        hidden_units = DB.getHiddenUnits( projectId, projectType )
        print 'hidden_units:', hidden_units
        hidden_units = [200]
        #DB.removeHiddenUnits( projectId )
        #DB.storeHiddenUnits( projectId, projectType, hidden_units )
        nKernels=[48,48]
        kernelSizes=[5,5]

        k = DB.getKernelSizes( projectId, projectType )
        n = DB.getNumKernels( projectId, projectType )
        print 'kernel sizes:', k
        print 'num kernels:', n

        DB.removeKernelSizes( projectId )
        DB.removeNumKernels( projectId )
        DB.storeKernels( projectId, projectType, nKernels, kernelSizes )

        hidden_units = DB.getHiddenUnits( projectId, projectType )
        print 'after:'
        print 'hidden_units:', hidden_units
        exit(1)

    # install mlp project
    p             = Project( id='felixmlp', type='MLP')
    p.batchSize   = 16
    p.patchSize   = 39
    p.hiddenUnits = [500,500,500]
    install( p )

    '''
    # install cnn project	
    cnn              = Project( id='felixcnn', type='CNN')
    cnn.trainTime    = 30
    cnn.learningRate = 0.01
    cnn.batchSize    = 128
    cnn.patchSize    = 39
    cnn.nKernels     = [48,48]
    cnn.kernelSizes  = [5,5]
    cnn.hiddenUnits  = [200]
    install( cnn )
    '''
