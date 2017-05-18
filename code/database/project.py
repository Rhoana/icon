#-------------------------------------------------------------------------------------------
# project.py
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

import os
import sqlite3 as lite
import sys
import json
import glob
import time
import uuid
from datetime import datetime, date

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))

from utility import Utility
from paths import Paths
from label import Label
from image import Image

class Project (object):

    CNN       = 'CNN'
    MLP       = 'MLP'
    UNET      = 'UNET'
    INVALID   = -1
    ONLINE    = 0
    OFFLINE   = 1
    TrainTime = 15  # 15 seconds
    SyncTime  = 20  # 20 seconds

    # create a new project object
    #----------------------------
    def __init__(self,
        id,
        type,
        revision=0,
        baseModel='',
        patchSize=39,
        batchSize=16,
        learningRate=0.01,
        momentum=0.9,
        hiddenUnits=[500,500,500],
        nKernels=[48,48],
        kernelSizes=[5,5],
        threshold=0.5,
        mean=0.5,
        data_mean=0.5,
        data_std=1.0
        ):

        self.activemode   = Project.ONLINE
        self.id           = id
        self.type         = type
        self.baseModel    = baseModel
        self.std          = data_std
        self.mean         = data_mean
        self.threshold    = threshold
        self.patchSize    = patchSize
        self.batchSize    = batchSize
        self.learningRate = learningRate
        self.momentum     = momentum
        self.hiddenUnits  = hiddenUnits
        self.nKernels     = nKernels
        self.kernelSizes  = kernelSizes
        self.trainTime    = Project.TrainTime
        self.syncTime     = Project.SyncTime
        self.epochs       = 100
        self.labels       = []
        self.images       = []
        self.validation_images = []

    def addLabel(self, index, name, r, g, b):
        self.labels.append( Label(index, name, r, g, b) )

    def addImage(self, imageId, annFile=None, segFile=None, score=0.0, purpose='train'):
        image = Image( imageId )
        image.purpose = purpose
        image.annotationFile = annFile
        image.segmentationFile = segFile
        image.traningScore = score
        self.images.append( image )

    def toJson(self):
        data = {}
        data['id']                          = self.id
        data['std']                         = self.std
        data['mean']                        = self.mean
        data['threshold']                   = self.threshold
        data['training_mod_status']         = self.trainingStatus
        data['training_mod_status_str']     = Project.statusToStr( self.trainingStatus )
        data['segmentation_mod_status']     = self.predictionStatus;
        data['segmentation_mod_status_str'] = Project.statusToStr( self.predictionStatus )
        data['initial_model']               = self.baseModel
        data['model_type']                  = self.type
        data['sample_size']                 = self.patchSize
        data['learning_rate']               = self.learningRate
        data['momentum']                    = self.momentum
        data['batch_size']                  = self.batchSize
        data['epochs']                      = self.epochs
        data['train_time']                  = self.trainTime
        data['sync_time']                   = self.syncTime
        data['model_mod_time']              = self.modelTime
        data['locked']                      = self.locked
        data['labels']                      = [l.toJson() for l in self.labels ]
        data['hidden_layers']               = json.dumps( self.hiddenUnits )
        data['num_kernels']                 = json.dumps( self.nKernels )
        data['kernel_sizes']                = json.dumps( self.kernelSizes )
        data['images']                      = [i.toJson() for i in self.images ]
        data['validation_images']           = [i.toJson() for i in self.validation_images ]
        data['offline']                     = self.offline
        data['online']                      = self.online
        data['baseline']                    = self.baseline
        data['stats']			            = [ s.toJson() for s in self.stats ]
        return data

    @staticmethod
    def fromJson(data):
        project              = Project(id=data['id'], type=data['model_type'])
        project.baseModel    = data['initial_model']
        project.std          = data['std']
        project.mean         = data['mean']
        project.threshold    = data['threshold']
        project.patchSize    = data['sample_size']
        project.batchSize    = data['batch_size']
        project.learningRate = data['learning_rate']
        project.momentum     = data['momentum']
        project.trainTime    = data['train_time']
        project.syncTime     = data['sync_time']
        print 'hidden_layers:', data['hidden_layers']
        print 'num_kernels:', data['num_kernels']
        print 'kernel_sizes:', data['kernel_sizes'], type(data['kernel_sizes'])
        project.hiddenUnits  = data['hidden_layers']  #json.loads( data['hidden_layers'] )
        project.nKernels     = data['num_kernels']  #jjson.loads( data['num_kernels'] )
        project.kernelSizes  = data['kernel_sizes']  #jjson.loads( data['kernel_sizes'] )
        return project

    def isTrainable(self):
        if len(self.labels) == 0:
            print 'no labels found...'
            return False

        if self.type == 'MLP':
            if len(self.hiddenUnits) == 0:
                print 'no hidden layer units found...'
                return False

        if self.type == 'CNN':
            if len(self.nKernels) == 0:
                print 'number of kernels not found...'
                return False
            elif len(self.kernelSizes) == 0:
                print 'kernel sizes not found...'
                return False
        return True

    @staticmethod
    def statusToStr( status ):
        if status == 1:
            return 'Active'
        elif status == 2:
            return 'Pending Annotations'
        else:
            return 'Inactive'
