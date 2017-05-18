#---------------------------------------------------------------------------
# model.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains common code for all learning models.
#---------------------------------------------------------------------------

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.initializations import uniform
from keras import backend as K
from generate_data import *
import theano
import multiprocessing
import sys
import matplotlib
import matplotlib.pyplot as plt

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../../external'))
sys.path.insert(2,os.path.join(base_path, '../../common'))
sys.path.insert(3,os.path.join(base_path, '../model'))
sys.path.insert(4,os.path.join(base_path, '../database'))

from db import DB
from utility import Utility
from paths import Paths

class IconModel(object):

    def __init__(self, project, offline=False):
        print 'Model()'
        self.done = False
        self.project = project
        self.offline = offline
        self.model = None
        self.best_val_loss_so_far = 0
        self.patience_counter = 0
        self.patience = 100
        self.patience_reset = 100

    def train(self):
        pass

    def predict(self, image):
        pass

    def save(self, best=False):
        print 'Model.save()'

        if self.model == None:
            return False

        name   = '%s_%s'%(self.project.id, self.project.type)
        prefix = 'best' if best else 'latest'
       
        revision = 0
 
        if self.offline:
            name = '%s_offline'%(name)
        elif best:
            revision = DB.getRevision( self.id )
            revision = (revision+1)%10
            prefix   = '%s_%d'%(prefix, revision)

        # construct the path to the network and weights
        path = '%s/%s_%s'%(Paths.Models, prefix, name)
        j_path = '%s.json'%(path)
        w_path = '%s_weights.h5'%(path)

        j_path = j_path.lower()
        w_path = w_path.lower()

        print 'saving model...'
        json_string = self.model.to_json()
        open(j_path, 'w').write(json_string)
        self.model.save_weights(w_path, overwrite=True)

        if not self.offline:
            DB.finishSaveModel( self.project.id, revision )

        return True

    def load(self, best=False):
        print 'Model.load()'

        name   = '%s_%s'%(self.project.id, self.project.type)
        prefix = 'best' if best else 'latest'

        if self.offline:
            name = '%s_offline'%(name)
        elif best:
            revision = DB.getRevision( self.id )
            prefix   = '%s_%d'%(prefix, revision)

        # construct the path to the network and weights
        path = '%s/%s_%s'%(Paths.Models, prefix, name)
        j_path = '%s.json'%(path)
        w_path = '%s_weights.h5'%(path)

        j_path = j_path.lower()
        w_path = w_path.lower()
       
        if not os.path.exists( j_path ) or not os.path.exists( w_path ):
            return False

        print 'loading model...'
        self.model = model_from_json(open( j_path ).read())
        self.model.load_weights( w_path )
        return True

    def threshold(self, prob, factor=0.5):
        prob[ prob >= factor ] = 9
        prob[ prob <  factor ] = 1
        prob[ prob == 9      ] = 0
        return prob


    def report_stats(self, elapsedTime, batchIndex, valLoss, trainCost):
        if not self.offline:
            DB.storeTrainingStats( self.project.id, valLoss, trainCost, mode=0)

        msg = '(%0.1f)     %i     %f%%'%\
        (
           elapsedTime,
           batchIndex,
           valLoss
        )
        status = '[%f]'%(trainCost)
        Utility.report_status( msg, status )


    @staticmethod
    def shared_dataset(data_xy, borrow=True, doCastLabels=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        if not doCastLabels:
            shared_y = theano.shared(np.asarray(data_y,
                            dtype=theano.config.floatX),
                            borrow=borrow)
        else:
            shared_y = theano.shared(np.asarray(data_y,
                            dtype=np.int32),
                            borrow=borrow)

        return shared_x, shared_y

