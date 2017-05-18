#---------------------------------------------------------------------------
# manager.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains the implementation of a thread for mananging the 
#           the execution of a DNN algorithm.  
#---------------------------------------------------------------------------

import os
import sys
import getopt
import signal
import threading
import time
import numpy as np

import theano
import theano.tensor as T

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))
sys.path.insert(2,os.path.join(base_path, './cnn'))
sys.path.insert(3,os.path.join(base_path, './mlp'))
sys.path.insert(4,os.path.join(base_path, './unet'))
sys.path.insert(5,os.path.join(base_path, '../database'))

from project import Project
from mlp import MLP 
#from mlp_offline import MLP_Offline
from cnn import CNN
#from unet import UNET
#from cnn_offline import CNN_Offline
from utility import Utility
from paths import Paths
from data import Data
from db import DB

#---------------------------------------------------------------------------
class Manager(threading.Thread):

    def __init__(self, name):
        threading.Thread.__init__(self)
        self.waittime    = 1.0

        # one per project
        self.name        = name
        self.online      = True
        self.projectId   = None
        self.model       = None
        self.dataset     = None
        self.done        = False
        self.x           = T.matrix('x')
        self.rng         = np.random.RandomState(929292)

    def can_load_model(self, path):
        return True

    def create_model(self, project):
        path = project.path
        self.model = None

        if not self.online:
            path = project.path_offline

        #if project.type != Project.UNET and  not self.can_load_model(path):
        #    return

        if project.type == Project.MLP:

                self.model = MLP(
                        id=project.id,
                        rng=self.rng,
                        input=self.x,
                        n_in=project.patchSize**2,
                        n_hidden=project.hiddenUnits,
                        n_out=len(project.labels),
                        train_time=project.trainTime,
                        batch_size=project.batchSize,
                        patch_size=project.patchSize,
                        path=path)

        elif project.type == Project.CNN:
                self.model = CNN(
                        rng=self.rng,
                        input=self.x,
                        batch_size=project.batchSize,
                        patch_size=project.patchSize,
                        nkerns=project.nKernels,
                        kernel_sizes=project.kernelSizes,
                        hidden_sizes=project.hiddenUnits,
                        train_time=project.trainTime,
                        momentum=project.momentum,
                        learning_rate=project.learningRate,
                        path=path,
                        id=project.id)

        #elif project.type == Project.UNET:
        #        self.model = UNET( project )

        self.dataset = Data( project )


    #-------------------------------------------------------------------
    # Main method of the threat - runs utnil self.done is false
    #-------------------------------------------------------------------
    def run(self):
        while not self.done: 
            # always query for the most current projects
            projects  = DB.getProjects()
            running   = None
            runningId = None

            # the first active project is the running project,
            # all others are deactivated 
            for project in projects:
                projectId     = project.id
                projectStatus = project.trainingStatus
                print project.startTime, project.id, projectStatus
                projectStatusStr = DB.projectStatusToStr( projectStatus )
                if projectStatus >= 1:
                    if running == None:
                        running = project
                        runningId = projectId
                    else:
                        print 'shutting down....', projectId
                        print projectStatusStr
                        print projectStatus
                        print runningId
                        DB.stopProject( projectId )
                        msg1 = 'stopping (%s)' %(projectId)
                        msg2 = '(%s) -> (Deactive)'%(projectStatusStr)
                        Utility.report_status( msg1, msg2 )

            # start the new project if changed.
            if self.projectId != runningId and running != None:
                print 'running: ', runningId, self.projectId 
                projectStatus = running.trainingStatus
                projectStatusStr = DB.projectStatusToStr( projectStatus )
                self.create_model( running )

                print 'created model...'
                print running.type
                print self.model
                # reset the training stats
                #if self.name == 'training':
                #	DB.removeTrainingStats( projectId )

                msg1 = 'starting (%s)' %(runningId)
                msg2 = '(Deactive) -> (Active)'
                Utility.report_status( msg1, msg2 )

            if self.model is not None:
                self.projectId = runningId
                self.work( running )
            time.sleep(self.waittime)
            print 'slept....', self.waittime

    #-------------------------------------------------------------------
    # Retrieve training trasks from database and call classifier to
    # perform actual training
    #-------------------------------------------------------------------
    def work(self, project):
        pass

    #-------------------------------------------------------------------
    # interface for online training or segmentation
    #-------------------------------------------------------------------
    def online(self, project):
        pass

    #-------------------------------------------------------------------
    # interface for offline training or segmentation
    #-------------------------------------------------------------------
    def offline(self, project):
        pass

    #-------------------------------------------------------------------
    # shuts down the application
    #-------------------------------------------------------------------
    def shutdown(self):
        Utility.report_status('shutting down %s manager'%(self.name), '')
        self.model.done = True
        self.done = True

    @staticmethod
    def start( argv, module):
        name        = argv[:1]
        usage       = 'usage: %s -m [online | offline] -p <project-name>'%(argv[0])
        options     = argv[1:]
        projectId   = None

        try:
                opts, args = getopt.getopt(options,'m:p:',['mode=','project='])
        except:
                pass

        for opt, arg in opts:
            if opt in ('-m', '--mode'):
                if arg == 'online':
                    module.online = True
                elif arg == 'offline':
                    module.online = False
            elif opt in ('-p', '--project'):
                projectId = arg


        if not module.online:
            if projectId == None:
                print usage
                exit(1)
        module.run()


