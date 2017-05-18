#---------------------------------------------------------------------------
# prediction_module.py
# 
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for 
#           Automatic Segmentation of Images
#
# Summary : This file contains the prediction module of the learning 
#           model.
#---------------------------------------------------------------------------

import os
import sys
import time
import ConfigParser
import pandas as pd
import numpy as np
import tifffile as tif
import theano
import theano.tensor as T
import cPickle


theano.config.floatX = 'float32'

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../external'))
sys.path.insert(2,os.path.join(base_path, '../../data/input/images'))
sys.path.insert(3,os.path.join(base_path, '../../data/output/weights'))
sys.path

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from mlp import MLP
from settings import Settings
from utility import Utility
from datasets import DataSets

class PredictionModule (object):

	#-------------------------------------------------------------------
	# Main constructor of PredictionModule
	#-------------------------------------------------------------------
	def __init__(self, img_id, settings):
		self.settings = settings		
		self.datasets = DataSets( img_id, settings )
		self.classifier = None


        def report(self, name, dims):
                msg = '%s'%(name)
                status = '['
                p_dim = None
                for dim in dims:
                        if p_dim == None:
                                status = '%s%s'%(status, dim)
                        else:
                                status = '%s x %s'%(status, dim)
                        p_dim  = dim
                status = '%s]'%status
                Utility.report_status( msg, status)

        #-------------------------------------------------------------------
        # Main constructor of MembraneModel
        # Initialize vari
        #-------------------------------------------------------------------
        def predict(self):
		Utility.report_status('starting prediction', '')

		msg = 'loading prediction data'
		if not self.datasets.load_test():
			Utility.report_status( msg, 'fail')
			return False

		Utility.report_status( msg, 'done' )

                self.report( 'test set x', \
                self.datasets.test_set_x.shape.eval())
                self.report( 'test set y', \
                self.datasets.test_set_y.shape.eval())

                path = '%s/best_model.pkl'%(self.settings.params)
                msg = 'loading best model'
                if not os.path.exists(path):
                        Utility.report_status( msg, 'fail' )
                        return False

                n_hidden = self.settings.n_hidden
                n_classes = self.datasets.n_classes
                n_features = self.datasets.n_features

		w_val = np.random.normal(0, 0.1, (n_features, n_hidden))
		b_val = np.zeros( n_hidden )
                w = theano.shared(value=w_val, name='W', borrow=True)
                b = theano.shared(value=b_val, name='b', borrow=True)
                save_file = open(path)
		w_val, b_val = cPickle.load(save_file)
                w.set_value(w_val)
		#cPickle.load(save_file), borrow=True)
                b.set_value(b_val)
		#cPickle.load(save_file), borrow=True)

		print 'w:', w.shape.eval()
		print 'b:', b.shape.eval()

                Utility.report_status( msg, 'done' )


		test_set_x = self.datasets.test_set_x
		test_set_y = self.datasets.test_set_y

		rng = np.random.RandomState(42)

                # Step 1. Declare Theano variables
                x = T.fmatrix()
                y = T.ivector()
                index = T.iscalar()

		path = '%s/%s'%(self.settings.models, 'best_model_mlp.pkl')
		print 'loading %'%(path)
		classifier = MLP_Ex( path )
		if True:
			return True

                classifier = MLP(
                        rng=rng,
                        input=x,
                        n_in=n_features,
                        n_hidden=n_hidden,
                        n_out=n_classes
                )   
                classifier.params = []
		classifier.params.extend([ w, b ])

                predict_model = theano.function(
                    inputs=[index],
                    outputs=classifier.logRegressionLayer.y_pred,
                    givens={
                            x: test_set_x,
                            y: test_set_y
                        },
		    on_unused_input='ignore' 
                    )   

		predicted_values = predict_model( 1 )
		print 'size predicted:', len( predicted_values )

		msg = 'image segmentation'
		Utility.report_status( msg, 'done' )
 		return True	
