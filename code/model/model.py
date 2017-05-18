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

import os
import sys
import time
import math
import numpy as np
import random

import theano
import theano.tensor as T

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../../external'))
sys.path.insert(2,os.path.join(base_path, '../../common'))
sys.path.insert(3,os.path.join(base_path, '../model'))
sys.path.insert(4,os.path.join(base_path, '../database'))

from utility import Utility
from utility import enum
from db import DB



class Model(object):

	Modes = enum( ONLINE=1, OFFLINE=2 )

	# Data split by percentage for
 	# [training, validation, testing]
	Split                  = [0.8, 0.10, 0.05]
	TrainMax               = 1024*5
	ValidMax               = 1024*2
	BestValidationInterval = 4

	def __init__(	self,
			rng,         # random number generator use to initialize weights
			input,       # a theano.tensor.dmatrix of shape (n_examples, n_in)
			batch_size,  # size of mini batch
			patch_size,  # size of feature map
			train_time,  # batch training time before resampling
			path,        # where to load or save best model
			type,        # type of model (CNN or MLP)
			id           # the unique identifier of the model (used for persistence)
			):

		self.done        = False
		self.id          = id
		self.type        = type
                self.path        = path
		self.train_time  = train_time
		self.batchSize   = batch_size
		self.patchSize   = patch_size
		self.input       = input
		self.rng         = rng
		self.params      = []
                self.best_loss   = np.inf
                self.i_train     = []
                self.i_valid     = []
                self.i_test      = []
                self.i           = []
		self.n_data      = 0
                self.iteration   = 0
		self.initialized = False
		self.x           =  input
		self.index       = T.lscalar()
		self.label_dist  = []

	def setTrainingData(self, 
		x,               #
		y,               #
		p,               #
		l,               #
		learning_rate,   #
		momentum):
		# save the training set and params
		self.x_data        = x
		self.y_data        = y
		self.p_data        = p
		self.i_labels      = l     
		self.n_data        = len(y)
		self.learning_rate = learning_rate
		self.momentum      = momentum	
		self.iteration     = 0

		# spli the training set into training, validation, and test
		self.n_train       = int(math.floor(self.n_data * Model.Split[0]))
		self.n_valid       = int(math.floor(self.n_data * Model.Split[1]))
		self.n_test        = int(math.floor(self.n_data * Model.Split[2]))
		self.n_train       = min(self.n_train, Model.TrainMax)

		self.n_valid       = Model.ValidMax
		self.n_train       = self.n_data - self.n_valid

		self.i             = np.arange( self.n_data )
		self.i             = np.random.choice( self.i, self.n_data, replace=False)

		self.i_valid       = self.i[self.n_train:self.n_train + self.n_valid]
		self.i_train       = []
		self.l_draws       = []
		l_percentages      = [ float(len(l_data))/self.n_data for l_data in l]
		self.n_draws       = [ int(self.n_train*l) for l in l_percentages]

		total = np.sum( self.n_draws )
		if (total < self.n_train):
			i_max = np.argmax( self.n_draws )
			self.n_draws[ i_max ] += (self.n_train - total)
		
		self.i_train_labels     = [ [] for l in l_percentages]
		self.i_valid_labels     = [ [] for l in l_percentages]

		#self.l_dist        = [ len(l_data) for l_data in l]
		'''
		print 'n_valid:', self.n_valid
		print '#valid:', len(self.i_valid)	
 		print '==setTrainingData'	
		print 'p_draws:',l_percentages
		print 'n_draws:', self.n_draws
		print 'total:', np.sum(self.n_draws)
		'''

	def drawRandomizedSamples(self):
		print '------randomized sampling-------'
		self.i_train = []
		for i, n_draw in enumerate(self.n_draws):
			self.i_labels[i]       = np.hstack( (self.i_labels[i], self.i_train_labels[i]) )
			n_ldata                = len(self.i_labels[i])
			self.i_labels[i]       = np.random.choice( self.i_labels[i], n_ldata, replace=False)
			self.i_train_labels[i] = self.i_labels[i][:n_draw]
			self.i_labels[i]       = self.i_labels[i][n_draw:]
			self.i_train           = np.concatenate( (self.i_train, self.i_train_labels[i]) )

	def drawStratifiedSamples(self):
		print '------stratified sampling-------'
		n_train      = len(self.i_train)
		self.i_train = []
		print 'n_train:', n_train
		for i, n_draw in enumerate(self.n_draws):
			if n_train > 0:
				# determine number of good and bad
				# samples based on the training results
				indices   = self.i_train_labels[i]
				n_indices = len(indices)
				p_train   = self.p_data[ indices ]
				i_sorted  = np.argsort( p_train, axis = 0)
				n_good    = len( np.where( p_train[ i_sorted ] == 0 )[0] )
				n_bad     = len( indices ) - n_good

				# keep at most 50% of the bad samples for
				# retraining, and resample the rest
				n_good    = max( n_good, n_draw/2)
				n_bad     = n_indices - n_good
				i_good    = i_sorted[ : n_good ]
				i_bad     = i_sorted[ n_good: ]

				print '-------', i, '---------'
				print 'n_indices:', n_indices
				print 'n_good:', n_good
				print 'n_bad:', n_bad
				print 'i_sorted:', i_sorted
				print 'indices:', indices
				print 'p_train:', len(p_train), p_train
				print 'i_sorted:', len(i_sorted), i_sorted

				# return the good indices back to the pool
				self.i_labels[i] = np.hstack( (self.i_labels[i], i_good))

				# draw replacement samples for the good indices
				i_new            = self.i_labels[i][:n_good]
				self.i_labels[i] = self.i_labels[i][n_good:]

				# combine with the bad indices to comprise the new 
				# training batch
				self.i_train_labels[i] = np.hstack( (i_new, i_bad) )
			else:
				self.i_train_labels[i] = self.i_labels[i][:n_draw]
				self.i_labels[i]       = self.i_labels[i][n_draw:]

			# add the label indices to the training batch
			self.i_train = np.concatenate( (self.i_train, self.i_train_labels[i]) )

	def rotateSamples(self):
		print 'rotateSamples....'
                # randomly rotate samples
		n_half  = len(self.i_train)/2
		indices = np.random.choice( self.i_train, n_half, replace=False)	

		for index in indices:
                        #print 'itrain:', self.i_train
                        #print 'index:', index
                        patch = self.x_data[ index ]
                        #print 'patch.a:',patch
                        patch = np.reshape( patch, (self.patchSize, self.patchSize))
			if random.random() < 0.5:
                        	patch = np.fliplr( patch )
			else:
				patch = np.fliplr( patch )
                        patch = patch.flatten()
                        #print 'patch.b:',patch
                        self.x_data[ index ] = patch
                        #exit(1)

	def justDraw(self):
		print 'just draw...'
		i = self.i[:self.n_train]
		self.i_train = np.random.choice( i, self.n_train, replace=False) 

		if (self.iteration%2 == 0):
			self.rotateSamples()

	def sampleTrainingData(self):
		
		self.iteration += 1

		'''	
                if self.iteration == Model.BestValidationInterval:
                        self.drawRandomizedSamples()
                else:
                        self.drawStratifiedSamples()
			#self.rotateSamples()
		'''

		self.justDraw()

		self.i_train = self.i_train.astype(dtype=int)

                lens = [len(l) for l in self.i_labels]
                print 'remaining sizes:', lens
                lens = [len(l) for l in self.i_train_labels]
                print 'ampleed sizes:', lens
                print 'train indices:', len(self.i_train)
                print self.i_train

		#t = self.i[:self.n_train]
		#self.i_train = np.random.choice( t, self.n_train, replace=False) 
		#self.i_valid = np.random.choice( self.i, self.n_valid, replace=False)
		
                train_x = self.x_data[ self.i_train ]
                train_y = self.y_data[ self.i_train ]
                valid_x = self.x_data[ self.i_valid ]
                valid_y = self.y_data[ self.i_valid ]
                test_x  = self.x_data[ self.i_test  ]
                test_y  = self.y_data[ self.i_test  ]

                print 'tx:',np.shape(train_x)
                print 'ty:',np.shape(train_y)
                print 'vx:',np.shape(valid_x)
                print 'vy:',np.shape(valid_y)

                if (self.iteration == Model.BestValidationInterval):
                        self.best_loss = np.inf
                        self.iteration = 0

                if self.initialized:
                        self.lr_shared.set_value( np.float32(self.learning_rate) )
                        self.m_shared.set_value( np.float32(self.momentum) )

                        self.train_x.set_value( np.float32( train_x ) )
                        self.valid_x.set_value( np.float32( valid_x ) )
                        self.test_x.set_value( np.float32( test_x ) )

                        self.train_y.owner.inputs[0].set_value( np.int32( train_y ))
                        self.valid_y.owner.inputs[0].set_value( np.int32( valid_y ))
                        self.test_y.owner.inputs[0].set_value( np.int32( test_y ))
                else:
                        self.y     = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
                        self.lr    = T.scalar('learning_rate')
                        self.m     = T.scalar('momentum')

                        self.lr_shared = theano.shared(np.float32(self.learning_rate))
                        self.m_shared  = theano.shared(np.float32(self.momentum))

                        self.train_x   = theano.shared( train_x, borrow=True)
                        self.valid_x   = theano.shared( valid_x, borrow=True)
                        self.test_x    = theano.shared( test_x, borrow=True)

                        self.train_y = theano.shared( train_y, borrow=True)
                        self.valid_y = theano.shared( valid_y, borrow=True)
                        self.test_y  = theano.shared( test_y, borrow=True)

                        self.train_y = T.cast( self.train_y, 'int32')
                        self.valid_y = T.cast( self.valid_y, 'int32')
                        self.test_y  = T.cast( self.test_y, 'int32')
                        self.intialized = True



	def sampleTrainingDataold(self):

		# iteration counter
		self.iteration += 1

		if self.iteration == Model.BestValidationInterval:
			self.drawRandomizedSamples()
		else:
			self.drawStratifiedSamples()

		if len(self.i_train) > 0:
			print 'second....'
			self.i_train = []
			if self.iteration == Model.BestValidationInterval:
				self.drawRandomizedSamples()
			else:
				self.drawStratifiedSamples()
	
			for i, n_draw in enumerate(self.n_draws):
				indices = self.i_train_labels[i]
				p_train = self.p_data[ indices ]	
				i_sorted = np.argsort( p_train, axis = 0)
				n_good   = len( np.where( p_train[ i_sorted ] == 0 )[0] )
				n_bad    = len( indices ) - n_good
                        	#n_good   = max( n_good, n_draw/2)
                        	#i_good   = i_sorted[ : n_good ]
                        	#i_bad    = i_sorted[ n_good: ]

			 	n_threshold = int(len(indices) * 0.30)


				print 'n_good:', n_good
				print 'n_bad:', n_bad
				print 'min:', n_threshold
				# if not enough bad samples, just pick random samples
				if self.iteration == Model.BestValidationInterval:
					print '--->random sampoing<----'
					self.i_labels[i]    = np.hstack( (self.i_labels[i], self.i_train_labels[i]) )
					n_ldata           = len(self.i_labels[i])
					print 'n_ldata:', n_ldata
					self.i_labels[i]    = np.random.choice( self.i_labels[i], n_ldata, replace=False)
                                	self.i_train_labels[i] = self.i_labels[i][:n_draw]
                                	self.i_labels[i]    = self.i_labels[i][n_draw:]
                                	self.i_train      = np.concatenate( (self.i_train, self.i_train_labels[i]) )
					self.drawRandomizedSamples()

				else:
					self.drawStratifiedSamples()
					print '--->5050 sampoing<-----'
					# keep 50% of the bad samples and replace the rest
					n_good   = max( n_good, n_draw/2)
					i_good   = i_sorted[ : n_good ]
					i_bad    = i_sorted[ n_good: ]

					self.i_labels[i] = np.hstack( (self.i_labels[i], i_good))
					i_new = self.i_labels[i][ : n_good ]
					self.i_train_labels[i] = np.hstack( (i_new, i_bad) )
					self.i_train = np.concatenate( (self.i_train, self.i_train_labels[i]) )
		else:
			#self.drawStratifiedSamples()
			self.i_train = []	
			self.i_train_labels = [[] for l in self.n_draws]

			for i, n_draw in enumerate(self.n_draws):
				self.i_train_labels[i] = self.i_labels[i][:n_draw]
				self.i_labels[i]    = self.i_labels[i][n_draw:]
				self.i_train      = np.concatenate( (self.i_train, self.i_train_labels[i]) )

				#i_draw = np.random.choice( self.i_labels[i], n_draw, replace=False )
				print i, n_draw, len(self.i_train_labels[i]), len(self.i_labels[i])

		self.i_train = self.i_train.astype(dtype=int)
		lens = [len(l) for l in self.i_labels]
		print 'remaining sizes:', lens
		lens = [len(l) for l in self.i_train_labels]
		print 'ampleed sizes:', lens
		print 'train indices:', len(self.i_train)
		print self.i_train
		#self.i       = self.i[ self.n_train: ]
		#self.i_train = self.i[ 0 : self.n_train ]
		#self.i       = self.i[ self.n_train: ]

		train_x = self.x_data[ self.i_train ]
		train_y = self.y_data[ self.i_train ]
		valid_x = self.x_data[ self.i_valid ]
		valid_y = self.y_data[ self.i_valid ]
		test_x  = self.x_data[ self.i_test  ]
		test_y  = self.y_data[ self.i_test  ]

		print 'tx:',np.shape(train_x)
		print 'ty:',np.shape(train_y)
		print 'vx:',np.shape(valid_x)
		print 'vy:',np.shape(valid_y)

		exit(1)

                if (self.iteration == Model.BestValidationInterval):
                        self.best_loss = np.inf
                        self.iteration = 0

                if self.initialized:
			print 'exiting here....'

                        self.lr_shared.set_value( np.float32(self.learning_rate) )
                        self.m_shared.set_value( np.float32(self.momentum) )

                        self.train_x.set_value( np.float32( train_x ) )
                        self.valid_x.set_value( np.float32( valid_x ) )
                        self.test_x.set_value( np.float32( test_x ) )

                        self.train_y.owner.inputs[0].set_value( np.int32( train_y ))
                        self.valid_y.owner.inputs[0].set_value( np.int32( valid_y ))
                        self.test_y.owner.inputs[0].set_value( np.int32( test_y ))
                else:

                        # allocate symbolic variables for the data
                        #self.index = T.lscalar()     # index to a [mini]batch
                        #self.x     = T.matrix('x')   # the data is presented as rasterized images
			#self.x     = self.input
                        self.y     = T.ivector('y')  # the labels are presented as 1D vector of [int] labels
                        self.lr    = T.scalar('learning_rate')
                        self.m     = T.scalar('momentum')

                        self.lr_shared = theano.shared(np.float32(self.learning_rate))
                        self.m_shared  = theano.shared(np.float32(self.momentum))

                        self.train_x   = theano.shared( train_x, borrow=True)
                        self.valid_x   = theano.shared( valid_x, borrow=True)
                        self.test_x    = theano.shared( test_x, borrow=True)

                        self.train_y = theano.shared( train_y, borrow=True)
                        self.valid_y = theano.shared( valid_y, borrow=True)
                        self.test_y  = theano.shared( test_y, borrow=True)

                	self.train_y = T.cast( self.train_y, 'int32')
                	self.valid_y = T.cast( self.valid_y, 'int32')
                	self.test_y  = T.cast( self.test_y, 'int32')
                        self.intialized = True

	def setupSegmentation(self):

		self.intialized = True


        def train(self):
                pass
        
        def classify(self, image):
                pass
        
        def predict(self, image):
                pass

	def save(self):
		print 'saving model...'
		pass

	def load(self):
		pass


        def reportTrainingStats(self, elapsedTime, batchIndex, valLoss, trainCost, mode=0):
                DB.storeTrainingStats( self.id, valLoss, trainCost, mode=mode)
                msg = '(%0.1f)     %i     %f%%'%\
                (
                   elapsedTime,
                   batchIndex,
                   valLoss
                )
                status = '[%f]'%(trainCost)
                Utility.report_status( msg, status )


