#---------------------------------------------------------------------------
# prediction_task.py
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
import threading
import time
import numpy as np

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))

from mlp_classifier import MLP_Classifier
from utility import Utility
from settings import Settings
from paths import Paths
from datasets import DataSets
from database import Database


#---------------------------------------------------------------------------
class Task(threading.Thread):

        #-------------------------------------------------------------------
        # Reads the input image and normalize it.  Also creates
        # a version of the input image with corner pixels
        # mirrored based on the sample size
        # arguments: - path : directory path where image is located
        #            - id   : the unique identifier of the image
        #            - pad  : the amount to pad the image by.
        # return   : returns original image and a padded version
        #-------------------------------------------------------------------
	def __init__(self, classifier, projectId, name, wait_time):
		threading.Thread.__init__(self)
		self.waittime = wait_time
		self.projectId = projectId
		self.done = False
		self.name = name
		self.dataset = None
		self.classifier = classifier
		self.classifier.model_path = Paths.Models
		self.classifier.segmentation_path = Paths.Segmentation
		self.classifier.project = self.projectId

        #-------------------------------------------------------------------
        # Reads the input image and normalize it.  Also creates
        # a version of the input image with corner pixels
        # mirrored based on the sample size
        # arguments: - path : directory path where image is located
        #            - id   : the unique identifier of the image
        #            - pad  : the amount to pad the image by.
        # return   : returns original image and a padded version
        #-------------------------------------------------------------------
	def run(self):
	    status = -1
	    while not self.done:
		project = Database.getProject( self.projectId )

		print 'tr: ', project['training_mod_status']
		print 'pr: ', project['segmentation_mod_status']

		s = status
	        if self.name == 'training':
			s = project['training_mod_status']
		elif self.name == 'prediction':
			s = project['segmentation_mod_status']

		if s != status:
			status = s
			statusStr = Database.projectStatusToStr( s )
                        Utility.report_status('%s (%s)' %(self.name, self.projectId), '(%s)'%(statusStr))

		print 'status: ', status
		if status != 0:
			self.work()
		time.sleep(self.waittime)

        #-------------------------------------------------------------------
        # Reads the input image and normalize it.  Also creates
        # a version of the input image with corner pixels
        # mirrored based on the sample size
        # arguments: - path : directory path where image is located
        #            - id   : the unique identifier of the image
        #            - pad  : the amount to pad the image by.
        # return   : returns original image and a padded version
        #-------------------------------------------------------------------
	def work(self):
		print 'working...'
		#pass

        #-------------------------------------------------------------------
        # Check if there's incoming data in the specified path
        # arguments: path - the path to check
        # return   : true if has text files, false otherwise
        #-------------------------------------------------------------------
	def abort(self):
		Utility.report_status('stopping', '%s (%s) task'%(self.name, self.projectId))
		self.done = True


#---------------------------------------------------------------------------
class PredictionTask (Task):

        #-------------------------------------------------------------------
        # Reads the input image and normalize it.  Also creates
        # a version of the input image with corner pixels
        # mirrored based on the sample size
        # arguments: - path : directory path where image is located
        #            - id   : the unique identifier of the image
        #            - pad  : the amount to pad the image by.
        # return   : returns original image and a padded version
        #-------------------------------------------------------------------
        def __init__(self, classifier, projectId):
                Task.__init__(self, classifier, projectId, 'prediction', 1.0)
		self.high = []
	 	self.low  = [] 
		self.model_mod_time = None
		self.priority = 0

        #-------------------------------------------------------------------
        # Reads the input image and normalize it.  Also creates
        # a version of the input image with corner pixels
        # mirrored based on the sample size
        # arguments: - path : directory path where image is located
        #            - id   : the unique identifier of the image
        #            - pad  : the amount to pad the image by.
        # return   : returns original image and a padded version
        #-------------------------------------------------------------------
        def work(self):

		if len(self.high) == 0:
			self.high = Database.getPredictionTasks( self.projectId, 1)

		if len(self.low) == 0:
			self.low = Database.getPredictionTasks( self.projectId, 0 )

		task = None
		if self.priority == 0 and len(self.high) > 0:
			self.priority = 1
			task = self.high[0]
			del self.high[0]
		elif len(self.low) > 0:
			self.priority = 0
			task = self.low[0]
			del self.low[0]

		if task == None:
			return

                project = Database.getProject( self.projectId )
		labels = Database.getLabels( self.projectId )

		imageId = task['image_id']
                model_mod_time = project['model_mod_time']
                sample_size = project['sample_size']
                n_hidden = [ h['units'] for h in project["hidden_layers"] ]
                labels = Database.getLabels( self.projectId )


		print 'predicting: ', imageId

		if self.model_mod_time != model_mod_time:
			self.classifier.reset()
			self.model_mod_time = model_mod_time

		self.classifier.predict(
			sample_size,
			len(labels),
			n_hidden,
			imageId,
			self.projectId,
			Paths.Images)

		Database.finishPredictionTask( self.projectId, imageId )


#---------------------------------------------------------------------------
class TrainingTask (Task):

        #-------------------------------------------------------------------
        # Reads the input image and normalize it.  Also creates
        # a version of the input image with corner pixels
        # mirrored based on the sample size
        # arguments: - path : directory path where image is located
        #            - id   : the unique identifier of the image
        #            - pad  : the amount to pad the image by.
        # return   : returns original image and a padded version
        #-------------------------------------------------------------------
	def __init__(self, classifier, projectId):
		Task.__init__(self, classifier, projectId, 'training', 2)
		self.dataset = DataSets(projectId)

        #-------------------------------------------------------------------
        # Reads the input image and normalize it.  Also creates
        # a version of the input image with corner pixels
        # mirrored based on the sample size
        # arguments: - path : directory path where image is located
        #            - id   : the unique identifier of the image
        #            - pad  : the amount to pad the image by.
        # return   : returns original image and a padded version
        #-------------------------------------------------------------------
	def work(self):

		print 'training......running....'

		project = Database.getProject( self.projectId )
		sample_size = project['sample_size']
		n_hidden = [ h['units'] for h in project["hidden_layers"] ]
		learning_rate = project['learning_rate']
		momentum = project['momentum']
		batch_size = project['batch_size']
		epochs = project['epochs']

		print '---------------------'
		print n_hidden
		print '---------------------'
		# no training until all labels are annotated
		labels = Database.getLabels( self.projectId )

		# must have training and hidden layer units to train
                if len(labels) == 0 or len(project['hidden_layers']) == 0:
			print 'no labels or hidden layers....'
                        return


		# check for new data
		#new_data = Database.hasTrainingTasks( self.projectId ) or not self.dataset.valid()
		#if new_data:
		new_data = self.dataset.load_training(sample_size)
			
		# cache the dataset
		if not self.dataset.valid():
			print 'invalid data....'
			return

		# train the classifier
		self.classifier.train( self.dataset.x, 
                                       self.dataset.y,
				       self.dataset.p,
				       new_data,
				       sample_size**2,
				       len(labels),
				       n_hidden,
				       learning_rate,
				       momentum,
				       batch_size,
				       epochs )

		# save statistics
		self.dataset.save_stats()
		#Database.finishTrainingTask( self.project )
