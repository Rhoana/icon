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

import os
import sqlite3 as lite
import sys
import json
import glob
import time
import uuid
from datetime import datetime, date
from utility import *
from paths import Paths
import traceback


base_path = os.path.dirname(__file__)
sys.path.insert(2,os.path.join(base_path, '../common'))
DATABASE_NAME = os.path.join(base_path, '../../data/database/icon.db')

class Tables:

    @staticmethod
    def drop():
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cur.execute("DROP TABLE IF EXISTS Image")
            cur.execute("DROP TABLE IF EXISTS Project")
            cur.execute("DROP TABLE IF EXISTS HiddenUnits")
            cur.execute("DROP TABLE IF EXISTS NumKernels")
            cur.execute("DROP TABLE IF EXISTS KernelSize")
            cur.execute("DROP TABLE IF EXISTS Label")
            cur.execute("DROP TABLE IF EXISTS TrainingStats")
            cur.execute("DROP TABLE IF EXISTS Performance")
            cur.execute("DROP VIEW IF EXISTS ImageView")

    @staticmethod
    def create():
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()

            # create liveconnect table for maintaining Interactive
            # requests from user and updates from running models
            # Purpose: 0=Training, 1=Validation, 2=Test
            cmd  = "CREATE TABLE IF NOT EXISTS Image"
            cmd += "("
            cmd += "ProjectId TEXT, "
            cmd += "ProjectType TEXT, "
            cmd += "ImageId TEXT, "
            cmd += "Purpose INT DEFAULT 3, "
            cmd += "SegmentationRequestTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "SegmentationTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "SegmentationPriority INT DEFAULT 0, "
            cmd += "TrainingTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "TrainingPriority INT DEFAULT 0, "
            cmd += "TrainingScore INT DEFAULT 0, "
            cmd += "AnnotationTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "AnnotationStatus INT DEFAULT 0, "
            cmd += "AnnotationLockId TEXT, "
            cmd += "AnnotationLockTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "AnnotationFile TEXT, "
            cmd += "SegmentationFile TEXT, "
            cmd += "PRIMARY KEY(ProjectId, ImageId)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table Image','done')


            # create trainingstats table for maintaining traiing stats
            cmd  = "CREATE TABLE IF NOT EXISTS TrainingStats"
            cmd += "("
            cmd += "ProjectId TEXT, "
            cmd += "ProjectMode INT, "
            cmd += "ValidationError REAL DEFAULT 0.0, "
            cmd += "TrainingCost REAL DEFAULT 0.0, "
            cmd += "TrainingTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "PRIMARY KEY(ProjectId, TrainingTime, ProjectMode)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table TrainingStats', 'done')


            # create project table for maintaining model settings
            cmd  = "CREATE TABLE IF NOT EXISTS Project"
            cmd += "("
            cmd += "Id TEXT, "
            cmd += "Type TEXT, "
            cmd += "Revision INT DEFAULT 0, "
            cmd += "BaseModel TEXT, "
            cmd += "TrainTime REAL, "
            cmd += "SyncTime REAL, "
            cmd += "Lock INT DEFAULT 0, "
            cmd += "PatchSize INT DEFAULT 39, "
            cmd += "LearningRate REAL DEFAULT 0.01, "
            cmd += "Momentum FLOAT DEFAULT 0.9, "
            cmd += "BatchSize INT DEFAULT 10, "
            cmd += "Epoch INT DEFAULT 20, "
            cmd += "ActiveMode INT DEFAULT 0, "
            cmd += "TrainingModuleStatus INT DEFAULT 0, "
            cmd += "PredictionModuleStatus INT DEFAULT 0, "
            cmd += "CreationTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "StartTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "ModelModifiedTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "PRIMARY KEY(Id, Type)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table Project', 'done')

            '''
            cmd  = "CREATE VIEW ModelView AS SELECT "
            cmd += "Id, p.Id, "
            cmd += "Type, m.Type, "
            cmd += "BaseProject, p.BaseProject, "
            cmd += "PatchSize, m.PatchSize, "
            cmd += "BatchSize, m.BatchSize, "
            cmd += "Momentum, m.Momentum, "
            cmd += "LearningRate, m.LearningRate, "
            cmd += "Epochs, m.Epochs, "
            cmd += "TrainingStatus, m.TrainingStatus, "
            cmd += "SegmentationStatus, m.SegmentationStatus, "
            cmd += "CreationTime, m.CreationTime, "
            cmd += "StartTime, m.StartTime, "
            cmd += "ModelModifiedTime, m.ModelModifiedTime, "
            cmd += "TrainTime, p.TrainTime, "
            cmd += "SyncTime, p.SyncTime, "
            cmd += "Lock, p.Lock, "
            cmd += "FROM Project p, Model m, WHERE m.ProjectId=p.Id "
            cur.execute( cmd )
            Utility.report_status('creating view ModelViewk', 'done')
            '''


            # create HiddenUnits table for maintaining hidden layer units.
            cmd  = "CREATE TABLE IF NOT EXISTS HiddenUnits"
            cmd += "("
            cmd += "ProjectId TEXT, "
            cmd += "ProjectType TEXT, "
            cmd += "Id INT, "
            cmd += "Units INT, "
            cmd += "PRIMARY KEY(ProjectId, Id)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table HiddenUnits', 'done')

            # create NumKernels table for maintaining number of kernels for CNN models
            cmd  = "CREATE TABLE IF NOT EXISTS NumKernels"
            cmd += "("
            cmd += "ProjectId TEXT, "
            cmd += "ProjectType TEXT, "
            cmd += "Id INT, "
            cmd += "Count INT, "
            cmd += "PRIMARY KEY(ProjectId, Id)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table NumKernels', 'done')

            # create KernelSize table for maintaining number of kernels for CNN models
            cmd  = "CREATE TABLE IF NOT EXISTS KernelSize"
            cmd += "("
            cmd += "ProjectId TEXT, "
            cmd += "ProjectType TEXT, "
            cmd += "Id INT, "
            cmd += "Size INT, "
            cmd += "PRIMARY KEY(ProjectId, Id)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table KernelSize', 'done')


            # create Label table for maintaining labels
            cmd  = "CREATE TABLE IF NOT EXISTS Label"
            cmd += "("
            cmd += "ProjectId TEXT, "
            cmd += "ProjectType TEXT, "
            cmd += "Id INT, "
            cmd += "Name TEXT, "
            cmd += "R INT, "
            cmd += "G INT, "
            cmd += "B INT, "
            cmd += "PRIMARY KEY(ProjectId, Id)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table Label', 'done')

            cmd  = "CREATE VIEW ImageView AS SELECT "
            cmd += "ProjectId, t.ProjectId, "
            cmd += "ImageId, t.ImageId, t.Purpose,  "
            cmd += "SegmentationTime, t.SegmentationTime, "
            cmd += "SegmentationPriority, t.SegmentationPriority, "
            cmd += "TrainingTime, t.TrainingTime, "
            cmd += "TrainingPriority, t.TrainingPriority, "
            cmd += "TrainingScore, t.TrainingScore, "
            cmd += "AnnotationTime, t.AnnotationTime, "
            cmd += "AnnotationLockTime, t.AnnotationLockTime, "
            cmd += "AnnotationStatus, t.AnnotationStatus, "
            cmd += "AnnotationLockId, t.AnnotationLockId, "
            cmd += "SegmentationRequestTime, t.SegmentationRequestTime, "
            cmd += "ModelModifiedTime, p.ModelModifiedTime, "
            cmd += "CreationTime, p.CreationTime, "
            cmd += "StartTime, p.StartTime, "
            cmd += "Lock, p.Lock, "
            cmd += "AnnotationFile, t.AnnotationFile, "
            cmd += "SegmentationFile, t.SegmentationFile "
            cmd += "FROM Image t, Project p WHERE t.ProjectId=p.Id "
            cur.execute( cmd )
            Utility.report_status('creating view ImageView', 'done')


            # create project table for maintaining model settings
            cmd  = "CREATE TABLE IF NOT EXISTS Performance"
            cmd += "("
            cmd += "ProjectId TEXT, "
            cmd += "Type TEXT, "
            cmd += "Threshold REAL DEFAULT 0.0, "
            cmd += "VariationInfo REAL DEFAULT 0.0, "
            cmd += "PixelError REAL DEFAULT 0.0, "
            cmd += "CreationTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "PRIMARY KEY(ProjectId,Type,Threshold)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table Performance', 'done')
