#---------------------------------------------------------------------------
# settings.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains a module that encapsulate the settings
#           of the model
#---------------------------------------------------------------------------

import os
import sys
import StringIO
import base64
import numpy as np;
import json

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../common'))

from utility import Utility
from database import Database
from paths import Paths

SETTINGS_JSON = '../../data/settings.json'
PATHS_JSON    = '../../data/paths.json'
PROJECTS_PATH = '../../data/projects'
TRAIN_TASK_TYPE    = 'train'
PREDICT_TASK_TYPE  = 'predict'
PREDICTED_TASK_TYPE = 'predicted'

#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------
class Settings:


	#NOTES
	# Prediction task names are stored in a json file named as follows:
	# <project>.predict.json
	#
	# Training task names are stored in a json file named as follows:
	# <project>.train.json

	#-------------------------------------------------------------------
	#
	#-------------------------------------------------------------------
	def __init__(self, settings):
		self.projectname    = settings['name']
		self.project        = settings['id']
		self.learning_rate  = settings["learning_rate"]
		self.epochs         = settings["epochs"]
		self.momentum       = settings["momentum"]
		self.batch_size     = settings["batch_size"]
		self.n_hidden       = [ h['units'] for h in settings["hidden_layers"] ]
		self.sample_size    = settings["sample_size"]
		self.model_type     = settings["model_type"]
		self.initial_model  = settings["initial_model"]
		self.labels         = dict( (label["name"], label["index"]) for label in settings["labels"] )

	@staticmethod
	def getTrainingTasks(projectId):
		return Settings.getTasks(projectId, TRAIN_TASK_TYPE)

	@staticmethod
	def addTrainingTask(projectId, taskId):
		Settings.addTask(projectId, taskId, TRAIN_TASK_TYPE)

	#@staticmethod
	#def removeTrainingTask(projectId, taskId):
	#	Settings.removeTask(projectId, taskId, TRAIN_TASK_TYPE)

	@staticmethod
	def getPredictionTasks(projectId):
		return Settings.getTasks(projectId, PREDICT_TASK_TYPE)

	@staticmethod
	def addPredictionTask(projectId, taskId):
		print 'addPredictionTask....'
		path = "%s/%s.%s.json"%(PROJECTS_PATH, projectId, PREDICT_TASK_TYPE)
		Utility.lock(path)
		Settings.reconcilePredictionTasks( projectId )
		Settings.addTask(projectId, taskId, PREDICT_TASK_TYPE)
		Utility.unlock(path)

	@staticmethod
	def getFinishedPredictionTasks(projectId):
		return Settings.getTasks(projectId, PREDICTED_TASK_TYPE)

	@staticmethod
	def addFinishedPredictionTask(projectId, taskId):
		Settings.addTask(projectId, taskId, PREDICTED_TASK_TYPE)

	@staticmethod
	def reconcilePredictionTasks(projectId):
		finishedTasks = Settings.getFinishedPredictionTasks( projectId )
		tasks = Settings.getPredictionTasks( projectId )
		print 'finishedTasks:', finishedTasks
		print 'tasks:', tasks
		for task in finishedTasks:
			print 'task:', task
			if task in tasks:
				print 'removing task:', task
				tasks.remove( task )
		path = "%s/%s.%s.json"%(PROJECTS_PATH, projectId, PREDICT_TASK_TYPE)
		Utility.saveJson( path, tasks )
		path = "%s/%s.%s.json"%(PROJECTS_PATH, projectId, PREDICTED_TASK_TYPE)
		Utility.saveJson( path, [] )

	@staticmethod
	def getTasks(projectId, taskType):
		path = "%s/%s.%s.json"%(PROJECTS_PATH, projectId, taskType)
		data = Utility.readJson( path )
		return [] if data == None else data

	@staticmethod
	def addTask(projectId, taskId, taskType):
		tasks = Settings.getTasks(projectId, taskType)
		if taskId in tasks:
			return
		tasks.append( taskId )
		path = "%s/%s.%s.json"%(PROJECTS_PATH, projectId, taskType)
		Utility.saveJson( path, tasks )

	@staticmethod
	def removeTask(projectId, taskId, taskType):
		tasks = Settings.getTasks(projectId, taskType)
		if taskId in tasks:
			tasks.remove( taskId)
		path = "%s/%s.%s.json"%(PROJECTS_PATH, projectId, taskType)
		if not Utility.isLocked( path ):
			Utility.saveJson( path, tasks )


	def getLabel(self, name):
		for label in self.labels:
			if label['name'] == name:
				return label
		return None

	def report(self):
	    Utility.report_status('learning rate', "%s"%self.learning_rate)
	    Utility.report_status('hidden units', "%s"%self.n_hidden)
	    Utility.report_status('sample_size', "%s"%self.sample_size)

	@staticmethod
	def get( project ):
		settings      = Settings.getall()
		return settings[ project ]

	@staticmethod
	def getall( ):
		projects = {}
		ps = Database.getProjects()
		for p in ps:
			projects[ p[ 'id' ] ] = Settings( p )
		return projects
