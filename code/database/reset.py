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

#---------------------------------------------------------------------------
# Entry point to the main function of the program.
#---------------------------------------------------------------------------
if __name__ == '__main__':

   
    projectIds = ['felixmlp', 'felixcnn', 'testmlp', 'testcnn', 'testmlpv2', 'testcnnv2'] 
    for id in projectIds:
        print 'reseting stats for id:', id
        DB.removeTrainingStats( id )
