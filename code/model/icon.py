#---------------------------------------------------------------------------
# Utility.py
#
# Author  : Felix Gonda
# Date    : July 10, 2015
# School  : Harvard University
#
# Project : Master Thesis
#           An Interactive Deep Learning Toolkit for
#           Automatic Segmentation of Images
#
# Summary : This file contains utility functions for reading, writing, and
#           processing images.
#---------------------------------------------------------------------------

import os
import sys
import threading
import time
import signal
import json
import glob

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../external'))
sys.path.insert(2,os.path.join(base_path, '../common'))

from mlp_classifier import MLP_Classifier
from tasks import TrainingTask
from tasks import PredictionTask
from utility import Utility
from database import Database

sys.path
from settings import Settings
from settings import Paths

INPUT_DIR = '../../data/images'
APP_NAME = 'icon (interactive connectomics)'


class Icon:

	#-------------------------------------------------------------------
	# construct the application and initalizes task list to empty
	#-------------------------------------------------------------------
	@staticmethod
	def install(self, mode):

        	# start in a blank slate
        	Database.reset()

        	# install the tables
        	Database.initialize()

        	# install the default model
        	project = 'test'
        	Database.storeLabel(project, 0, 'background', 255,0,0)
        	Database.storeLabel(project, 1, 'membrane', 0,255,0)
        	Database.storeProject(project, project, '', 'MLP', 39, 0.01, 0.9, 20, 20, 15, 20, False)
        	Database.storeHiddenLayerUnit(project, 0, 500)
        	Database.storeHiddenLayerUnit(project, 1, 120)
        	Database.storeHiddenLayerUnit(project, 2, 100)


#---------------------------------------------------------------------------
# Entry point to the main function of the program.
#---------------------------------------------------------------------------
if __name__ == '__main__':
	signal.signal(signal.SIGINT, signal_handler)
	if len(sys.argv) <= 1 or sys.argv[1] not in ['train', 'predict', 'setup']:
		print 'Usage: python icon.py <train | predict | setup>'
	else:
		icon = Icon( sys.argv[ 1 ] )
		icon.run()
