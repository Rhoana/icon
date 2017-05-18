#---------------------------------------------------------------------------
# train.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains the implementation of a prediction task
#           thread.  The prediction task is responsible for segmenting
#           an entire image based a trained model.  It loads the trained
#           model from file.
#---------------------------------------------------------------------------

import os
import sys
import signal
import threading
import time
import numpy as np
import mahotas
import theano
import theano.tensor as T

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))
sys.path.insert(2,os.path.join(base_path, '../database'))

from manager import Manager
from utility import Utility
from paths import Paths
from db import DB

#---------------------------------------------------------------------------
class Training(Manager):

    #-------------------------------------------------------------------
    # Reads the input image and normalize it.  Also creates
    # a version of the input image with corner pixels
    # mirrored based on the sample size
    # arguments: - path : directory path where image is located
    #            - id   : the unique identifier of the image
    #            - pad  : the amount to pad the image by.
    # return   : returns original image and a padded version
    #-------------------------------------------------------------------
    def __init__(self):
        Manager.__init__( self, 'training' )
        self.version = 0

    #-------------------------------------------------------------------
    # online training
    #-------------------------------------------------------------------
    def work(self, project):
        print 'Training.work...'
        if project == None:
            return

        # Offline training
        offline = not self.online
        if offline or project.type == 'UNET':
            self.model.path = project.path_offline
            self.model.train(offline=offline, mean=project.mean, std=project.std)
            self.done = True
            return

        # Online training
        if project is None:
            #print 'no project...'
            return

        print 'trainable:', project.isTrainable()
        #print 'training......running....'
        if not project.isTrainable():
            return

        # check for new data
        '''
        if self.dataset.load( project ):
            self.model.setTrainingData( 
                self.dataset.x,
                self.dataset.y,
                self.dataset.p,
                self.dataset.l,
                project.learningRate,
                project.momentum)
        '''

        self.dataset.load( project )

        #self.dataset.sample()

        # cache the dataset
        if not self.dataset.valid():
            print 'invalid data...'
            return

        # train the classifier
        self.model.train( offline=False, data=self.dataset, mean=project.mean, std=project.std)

        #self.test_perf( project )

        # save statistics
        #self.dataset.save_stats( project )

        print 'done:', self.done


    def test_perf(self, project):
        name = 'train-input_0037.tif'
        path = '%s/%s'%(Paths.TrainGrayscale, name)
        image = mahotas.imread( path )
        image = Utility.normalizeImage( image )
        results = self.model.predict( image=image, mean=project.mean, std=project.std, threshold=project.threshold)
        n_membrane = len(results[ results == 1 ])
       
        print 'n_membrane:', n_membrane
 
        if n_membrane > 300000:
            rev  = DB.getRevision( self.model.id )
            version = '%d_%d'%(self.version, n_membrane)
            self.model.save_t( version )
            self.version += 1
                    

manager = None
def signal_handler(signal, frame):
        if manager is not None:
                manager.shutdown()

if __name__ == '__main__':
    Utility.report_status('running training manager', '')
    signal.signal(signal.SIGINT, signal_handler)
    manager = Training( )

    print 'manager:', manager
    Manager.start( sys.argv, manager )

