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
from utility import *
from paths import Paths
import traceback


base_path = os.path.dirname(__file__)
DATABASE_NAME = os.path.join(base_path, '../../data/database/icon.db')


class Database:

    @staticmethod
    def reset():
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cur.execute("DROP TABLE IF EXISTS Task")
            cur.execute("DROP TABLE IF EXISTS Project")
            cur.execute("DROP TABLE IF EXISTS HiddenUnits")
            cur.execute("DROP TABLE IF EXISTS Label")
	    cur.execute("DROP TABLE IF EXISTS Batch")
            cur.execute("DROP TABLE IF EXISTS Performance")
	    cur.execute("DROP VIEW IF EXISTS TaskView")

    @staticmethod
    def initialize():
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()

            # create liveconnect table for maintaining Interactive
            # requests from user and updates from running models
            cmd  = "CREATE TABLE IF NOT EXISTS Task"
            cmd += "("
            cmd += "ProjectId TEXT, "
            cmd += "ImageId TEXT, "
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
            Utility.report_status('creating table Task','done')

            # create Batch table for maintaining model settings
            cmd  = "CREATE TABLE IF NOT EXISTS Batch"
            cmd += "("
	    cmd += "Id TEXT, "
            cmd += "ProjectId TEXT, "
	    cmd += "ProjectType TEXT, "
            cmd += "ValidationError REAL DEFAULT 0.0, "
            cmd += "AverageCost REAL DEFAULT 0.0, "
            cmd += "TrainingTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "PRIMARY KEY(ProjectId, ProjectType, Id)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table Batch', 'done')


            # create project table for maintaining model settings
            cmd  = "CREATE TABLE IF NOT EXISTS Project"
            cmd += "("
            cmd += "Id TEXT, "
            cmd += "TrainingModuleStatus INT DEFAULT 0, "
            cmd += "PredictionModuleStatus INT DEFAULT 0, "
            cmd += "BaseModel TEXT, "
            cmd += "Type TEXT, "
            cmd += "PatchSize INT, "
            cmd += "LearningRate REAL, "
            cmd += "Momentum FLOAT, "
            cmd += "BatchSize INT, "
            cmd += "Epoch INT, "
            cmd += "TrainTime REAL, "
            cmd += "SyncTime REAL, "
	    cmd += "CreationTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
	    cmd += "StartTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "ModelModifiedTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
	    cmd += "Lock INT DEFAULT 0, "
            cmd += "PRIMARY KEY(Id)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table Project', 'done')

            # create HiddenUnits table for maintaining hidden layer units.
            cmd  = "CREATE TABLE IF NOT EXISTS HiddenUnits"
            cmd += "("
            cmd += "ProjectId TEXT, Id INT, Units INT, "
            cmd += "PRIMARY KEY(ProjectId, Id)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table HiddenUnits', 'done')

            # create Label table for maintaining labels
            cmd  = "CREATE TABLE IF NOT EXISTS Label"
            cmd += "("
            cmd += "ProjectId TEXT, Id INT, Name TEXT, "
            cmd += "R INT, G INT, B INT, "
            cmd += "PRIMARY KEY(ProjectId, Id)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table Label', 'done')

	    cmd  = "CREATE VIEW TaskView AS SELECT "
	    cmd += "ProjectId, t.ProjectId, "
	    cmd += "ImageId, t.ImageId, "
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
	    cmd += "Lock, p.Lock "
	    cmd += "FROM Task t, Project p WHERE t.ProjectId=p.Id "
	    cur.execute( cmd )
            Utility.report_status('creating view PredictionTask', 'done')


           # create project table for maintaining model settings
            cmd  = "CREATE TABLE IF NOT EXISTS Performance"
            cmd += "("
            cmd += "ProjectId TEXT, "
            cmd += "Type TEXT, "
            cmd += "Threshold REAL DEFAULT 0.0, "
            cmd += "VariationInfo REAL DEFAULT 0.0, "
            cmd += "PixelError REAL DEFAULT 0.0, "
            cmd += "CreationTime TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            cmd += "PRIMARY KEY(ProjectId,Threshold,Type)"
            cmd += ")"
            cur.execute( cmd )
            Utility.report_status('creating table Project', 'done')

	    #create view empdept as select empid, e.name, title, d.name, location from employee e, dept d where e.deptid = d.deptid;


    #---------------------------------------------------------------------------------------------
    # save model performance data
    #---------------------------------------------------------------------------------------------
    @staticmethod
    def storeModelPerformance(projectId, perfType, threshold, variationInfo, pixelError):
	#print 'storing...',projectId, perfType, threshold, variationInfo, pixelError
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "INSERT OR REPLACE INTO "
            cmd += "Performance(ProjectId, Type, Threshold, VariationInfo, PixelError, CreationTime)"
            cmd += "VALUES(?,?,?,?,?,?)"
            vals = (projectId, perfType, threshold, variationInfo, pixelError, now)
            cur.execute( cmd, vals )


    #---------------------------------------------------------------------------------------------
    # returns a list of performances results for a given project and type
    #---------------------------------------------------------------------------------------------
    @staticmethod
    def getPerformance(projectId, perfType):
        perf = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Threshold, VariationInfo, PixelError, CreationTime FROM Performance "
            cmd += "WHERE ProjectId=? AND Type=?"
	    vals = (projectId, perfType)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            for result in results:
                batch = {}
                batch['threshold'] = result[0]
                batch['variation_info'] = result[1]
                batch['pixel_error'] = result[2]
                batch['creation_time'] = result[3]
                perf.append( batch )
        return perf

    @staticmethod
    def getAll(projectId):
        perf = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Threshold, VariationInfo, PixelError, CreationTime, Type FROM Performance "
            cmd += "WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                batch = {}
                batch['threshold'] = result[0]
                batch['variation_info'] = result[1]
                batch['pixel_error'] = result[2]
                batch['creation_time'] = result[3]
		batch['type'] = result[4]
                perf.append( batch )
        return perf


    @staticmethod
    def getOfflinePerformance(projectId):
        return Database.getPerformance(projectId, 'offline')

    @staticmethod
    def getOnlinePerformance(projectId):
        return Database.getPerformance(projectId, 'online')

    @staticmethod
    def getBaselinePerformance(projectId):
        return Database.getPerformance(projectId, 'baseline')

    #---------------------------------------------------------------------------------------------
    # save batch training status
    #---------------------------------------------------------------------------------------------
    @staticmethod
    def storeBatch(projectId, projectType, batchId, valLoss, avgCost):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "INSERT OR REPLACE INTO "
	    cmd += "Batch(Id, ProjectId, ProjectType, "
	    cmd += "ValidationError, AverageCost, TrainingTime )"
            cmd += "VALUES(?,?,?,?,?)"
            vals = (batchId, projectId, projectType, valLoss, avgCost, now)
            cur.execute( cmd, vals )

    #---------------------------------------------------------------------------------------------
    # returns a list of batch status for the specified project
    #---------------------------------------------------------------------------------------------
    @staticmethod
    def getTrainingBatches(projectId, projectType):
        batches = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Id, ValidationError, AverageCost, TrainingTime"
            cmd += "WHERE ProjectId=? AND ProjectType=?"
	    vals = (projectId, projectType)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                batch = {}
		batch['id'] = result[0]
		batch['validation_error'] = result[1]
		batch['average_cost'] = result[2]
		batch['train_time'] = result[3]
                batches.append( batch )
        return batches

    #---------------------------------------------------------------------------------------------
    # Task table operations
    #---------------------------------------------------------------------------------------------
    @staticmethod
    def hasTrainingTasks(projectId):
	connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
	    cmd  = "SELECT COUNT(*) FROM TaskView WHERE "
	    cmd += "ProjectId='%s' ORDER BY SegmentationRequestTime DESC"%(projectId)
            cur.execute( cmd )
            results = cur.fetchone()[0]
	    return (results != 0)
	return False

    @staticmethod
    def finishLoadingTrainingTask( projectId, imageId ):
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur  = connection.cursor()
            cmd  = "UPDATE Task SET TrainingPriority = 0, TrainingTime=? "
	    cmd += "WHERE ProjectId=? AND ImageId=?"
	    vals = (now, projectId, imageId)
            cur.execute( cmd, vals )
	    connection.commit()

    @staticmethod
    def finishLoadingTrainingTasks( projectId ):
	Database.updateTrainingModuleStatus(projectId, 1)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur  = connection.cursor()
            cmd  = "UPDATE Task SET TrainingPriority = 0, TrainingTime=? "
            cmd += "WHERE ProjectId=?"
	    vals = (now, projectId)
            cur.execute( cmd, vals )
            connection.commit()

    @staticmethod
    def finishPredictionTask( projectId, imageId, duration ):
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	segFile = '%s/%s.%s.seg'%(Paths.Segmentation, imageId, projectId)

	print 'finishPredictionTask:', now
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd = "UPDATE Task SET SegmentationTime=?, SegmentationFile=?, SegmentationPriority=0 WHERE ProjectId=? AND ImageId=?"
            val = (now, segFile, projectId, imageId)
            cur.execute( cmd, val )

            cmd  = "UPDATE Project SET SyncTime=? WHERE Id=?"
            vals = (duration, projectId)
            cur.execute( cmd, vals )
	    connection.commit()

    @staticmethod
    def requestSegmentation(projectId, imageId):
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task = Database.getTask(projectId, imageId)
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
	    cmd  = "UPDATE Task SET SegmentationPriority=1, SegmentationRequestTime=? "
	    cmd += "WHERE ProjectId=? AND ImageId=?"
	    vals = (now, projectId, imageId)
	    cur.execute( cmd, vals )

    @staticmethod
    def storeTaskScore(projectId, imageId, score):
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Task SET TrainingScore=? WHERE ProjectId=? AND ImageId=?"
            vals = (score, projectId, imageId)
            cur.execute( cmd, vals )


    @staticmethod
    def saveAnnotations(projectId, imageId, annFile):
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        task = Database.getTask(projectId, imageId)
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Task SET TrainingPriority=1, SegmentationPriority=1, AnnotationFile=?, AnnotationTime=? "
            cmd += "WHERE ProjectId=? AND ImageId=?"
	    vals = (annFile, now, projectId, imageId)
	    cur.execute( cmd, vals )

    @staticmethod
    def storeTask(projectId, imageId, annFile, segFile):
	now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT * FROM Task WHERE ProjectId=? AND ImageId=?"
            vals = (projectId, imageId)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            if len(results) > 0:
                cmd  = "UPDATE Task SET AnnotationFile=?, SegmentationFile=?, AnnotationTime=? "
                cmd += "WHERE ProjectId=? AND ImageId=?"
		vals = (annFile, segFile, now, projectId, imageId)
            else:
            	cmd  = "INSERT INTO Task(AnnotationFile, SegmentationFile, ProjectId, ImageId, "
	    	cmd += "SegmentationTime, TrainingTime, AnnotationTime, SegmentationRequestTime) "
            	cmd += "VALUES(?,?,?,?,?,?,?,?)"
            	vals = (annFile, segFile, projectId, imageId, now, now, now, now)
            cur.execute( cmd, vals )

    @staticmethod
    def removeTask(projectId, imageId):
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "DELETE FROM Task WHERE ProjectId=? AND ImageId=?"
            vals = (projectId, imageId)
            cur.execute( cmd, vals )

    @staticmethod
    def getTrainingTasks(projectId, new):
        tasks = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT ImageId FROM Task "

	    if new:
            	cmd += "WHERE AnnotationFile is NOT NULL AND TrainingTime < AnnotationTime "
            else:
		cmd += "WHERE AnnotationFile is NOT NULL "

	    cmd += "AND ProjectId='%s' ORDER BY AnnotationTime DESC"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                task = {}
                task['image_id'] = result[0]
                tasks.append( task )
        return tasks



    @staticmethod
    def getPredictionTasks(projectId, priority=0):
        tasks = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
	    cmd  = "SELECT ImageId, SegmentationTime, ModelModifiedTime, Lock, "
	    cmd += "SegmentationRequestTime FROM TaskView "
	    cmd += "WHERE Lock == 0 AND SegmentationTime < ModelModifiedTime "
            cmd += "AND ProjectId=? AND SegmentationPriority=? ORDER BY SegmentationRequestTime DESC"
            vals = (projectId, priority)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            for result in results:
                task = {}
                task['image_id'] = result[0]
		task['seg_time'] = result[1]
		task['mod_time'] = result[2]
		task['lock'] = result[3]
		task['req_time'] = result[4]
                tasks.append( task )
        return tasks


    @staticmethod
    def lockTask(projectId, imageId):
	guid = uuid.uuid1().hex
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	connection = lite.connect( DATABASE_NAME )
	with connection:
		cur = connection.cursor()
		cmd  = "UPDATE Task SET AnnotationStatus=1, AnnotationLockId=?, "
		cmd += "AnnotationLockTime=? WHERE ProjectId=? AND ImageId=?"
		vals = (guid, now, projectId, imageId)
		cur.execute( cmd, vals )
	return guid

    @staticmethod
    def refreshAnnotationLock(projectId, imageId, guid):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME )
        with connection:
                cur = connection.cursor()
                #cmd = "UPDATE Task SET AnnotationStatus=1, AnnotationTime=? WHERE AnnotationLockId=? AND ProjectId=? AND ImageId=?"
		cmd = "UPDATE Task SET AnnotationLockTime=? WHERE AnnotationLockId=? AND ProjectId=? AND ImageId=?"
                vals = (now, guid, projectId, imageId)
                cur.execute( cmd, vals )
        return guid

    @staticmethod
    def getTask(projectId, imageId):
        task = {}
        task['project_id'] = projectId
        task['image_id'] = imageId
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT TrainingTime, TrainingPriority, TrainingScore, "
            cmd += "SegmentationTime, SegmentationPriority, SegmentationRequestTime, "
            cmd += "AnnotationTime, AnnotationStatus, AnnotationLockId, "
            cmd += "SegmentationRequestTime, ModelModifiedTime, "
	    cmd += "CreationTime, StartTime "
            cmd += "FROM TaskView WHERE ProjectId=? AND ImageId=?"
            vals = (projectId, imageId)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            if len(results) > 0:
                result = results[0]
                task["training_time"] = result[0]
                task["training_status"] = result[1]
                task["training_score"] = result[2]
                task["segmentation_time"] = result[3]
                task["segmentation_status"] = result[4]
		task["segmentation_req_time"] = result[5]
                task["annotation_time"] = result[6]
                task["annotation_status"] = result[7]
	   	task["annotation_lockid"] = result[8]
                task['pred_req_time'] = result[9]
		task['model_mod_time'] = result[10]
		task['creation_time'] = result[11]
		task['start_time'] = result[12]

                # set a flag when new segmentation is available
		creationTime = time.strptime(task['creation_time'], '%Y-%m-%d %H:%M:%S')
		startTime = time.strptime(task['start_time'], '%Y-%m-%d %H:%M:%S')
                segTime = time.strptime(task["segmentation_time"], '%Y-%m-%d %H:%M:%S')
                modelTime = time.strptime(task["model_mod_time"], '%Y-%m-%d %H:%M:%S')
		task['has_new_model'] = segTime < modelTime and startTime > creationTime
		'''
		creationTime = datetime.datetime.strptime(task['creation_time'], '%Y-%m-%d %H:%M:%S')
		startTime = datetime.datetime.strptime(task['start_time'], '%Y-%m-%d %H:%M:%S')
		segTime = datetime.datetime.strptime(task["segmentation_time"], '%Y-%m-%d %H:%M:%S')
		modelTime = datetime.datetime.strptime(task["model_mod_time"], '%Y-%m-%d %H:%M:%S')
		task['has_new_model'] = segTime < modelTime and startTime > creationTime
		print 'ct:', task['creation_time']
		print 'st:', task['start_time']
		print 'sgt:', task["segmentation_time"]
		print 'mt:', task["model_mod_time"]
		print 'segTime < modelTime: ', segTime < modelTime
		print 'startTime > creationTime: ', startTime > creationTime
                task['has_new_model'] = segTime < modelTime and startTime > creationTime
		print task['has_new_model']
		print '---'
		'''
        return task

    @staticmethod
    def getTaskStatus(projectId, imageId):
        data = {}
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cmd  = "SELECT SegmentationTime, SegmentationPriority, "
            cmd += "TrainingTime, TrainingPriority, TrainingScore, "
            cmd += "AnnotationTime, AnnotationStatus, AnnotationLockId "
            cmd += "FROM Task WHERE ProjectId=? AND ImageId=?"
            vals = (projectId, imageId)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            if len(results) > 0:
                result = results[0]
                data["segmentation_time"] = results[0]
                data["segmentation_status"] = results[1]
                data["training_time"] = results[2]
                data["training_status"] = results[3]
                data["training_score"] = results[4]
                data["annotation_time"] = results[5]
                data["annotation_status"] = results[6]
		data["annotation_lockid"] = results[7]
        return data


    @staticmethod
    def getTasks(projectId):
        tasks = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT ProjectId, ImageId, AnnotationFile, SegmentationFile, "
            cmd += "TrainingScore FROM Task "
            cmd += "WHERE ProjectId='%s' ORDER BY TrainingScore DESC"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                task = {}
                task['project_id'] = result[0]
                task['image_id'] = result[1]
                task['ann_file'] = result[2]
                task['seg_file'] = result[3]
                task['score'] = result[4]
                tasks.append( task )
        return tasks

    @staticmethod
    def getTaskNames(projectId):
        names = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT ImageId FROM Task WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                names.append(result[0])
        return names

    #---------------------------------------------------------------------------------------------
    # Label table operations
    #---------------------------------------------------------------------------------------------
    @staticmethod
    def storeLabel(projectId, index, name, r, g, b ):
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "INSERT OR REPLACE INTO Label(ProjectId, Id, Name, R, G, B)"
            cmd += "VALUES (?,?,?,?,?,?)"
            vals = (projectId, index, name, r, g, b)
            cur.execute( cmd, vals )

    @staticmethod
    def getLabels(projectId):
        labels = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT * from Label WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                label = {}
                label['index'] = result[1]
                label['name'] = result[2]
                label['r'] = result[3]
                label['g'] = result[4]
                label['b'] = result[5]
                labels.append( label )
        return labels

    @staticmethod
    def getLabel(projectId, name):
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Id, Name, R, G, B FROM Label WHERE ProjectId='%s' AND Name="%(projectId, name)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                label = {}
                label['index'] = result[0]
                label['name'] = result[1]
                label['r'] = result[2]
                label['g'] = result[3]
                label['b'] = result[4]
		return label
        return None


    #---------------------------------------------------------------------------------------------
    # Project table operations
    #---------------------------------------------------------------------------------------------
    @staticmethod
    def addProject( project ):
        connection = lite.connect( DATABASE_NAME )
        with connection:
		cur  = connection.cursor()
            	cmd  = "INSERT OR REPLACE INTO Project("
                cmd += "Id, BaseModel, Type, PatchSize, LearningRate, "
                cmd += "Momentum, BatchSize, Epoch, TrainTime, SyncTime, "
                cmd += "TrainingModuleStatus, PredictionModuleStatus, ModelModifiedTime, "
                cmd += "CreationTime )"
                cmd += "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"

                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                vals = (project['id'],
                	project['initial_model'],
                	project['model_type'],
                	project['sample_size'],
                	project['learning_rate'],
                	project['momentum'],
                	project['batch_size'],
                	project['epochs'],
			project['train_time'],
			project['sync_time'],
			0,
			0,
                        now,
                        now)
		cur.execute( cmd, vals )


    @staticmethod
    def removeProject(projectId):
        project = None
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "DELETE from Project WHERE Id='%s'"%(projectId)
            cur.execute( cmd )

        with connection:
            cur = connection.cursor()
            cmd  = "DELETE from Task WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )

	with connection:
	    cur = connection.cursor()
	    cmd = "DELETE FROM HiddenUnits WHERE ProjectId='%s'"%(projectId)
	    cur.execute( cmd )

        with connection:
            cur = connection.cursor()
            cmd = "DELETE FROM Label WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )


    @staticmethod
    def stopProject(projectId):
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Project SET TrainingModuleStatus=0, "
            cmd += "PredictionModuleStatus=0 WHERE Id='%s'"%(projectId)
            cur.execute( cmd )

    @staticmethod
    def startProject(projectId):
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Project SET TrainingModuleStatus=1, PredictionModuleStatus=1, "
            cmd += "StartTime=? WHERE Id=?"
	    vals = (now, projectId)
            cur.execute( cmd, vals )


    @staticmethod
    def isStopped(projectId):
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT TrainingModuleStatus FROM Project WHERE Id='%s'"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            if len(results) > 0:
                return (results[0][0] == 1)
        return False

    @staticmethod
    def updateTrainingModuleStatus(projectId, status):
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Project SET TrainingModuleStatus=? WHERE Id=?"
            vals = (status, projectId)
            cur.execute( cmd, vals )

    @staticmethod
    def updatePredictionModuleStatus(projectId, status):
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Project SET PredictionModuleStatus=? WHERE Id=?"
            vals = (status, projectId)
            cur.execute( cmd, vals )

    @staticmethod
    def getActiveProject():
  	project = None
	connection = lite.connect( DATABASE_NAME )
	with connection:
	    cur = connection.cursor()
	    cmd  = "SELECT Id, TrainingModuleStatus, PredictionModuleStatus, "
            cmd += "BaseModel, Type, PatchSize, LearningRate, "
            cmd += "Momentum, BatchSize, Epoch, TrainTime, SyncTime, ModelModifiedTime, Lock "
            cmd += "FROM Project WHERE TrainingModuleStatus > 0"
	    cur.execute( cmd )
	    results = cur.fetchall()
	    if len(results) > 0:
                result  = results[0]
                project = {}
                project["id"] = result[0]
                project['training_mod_status'] = result[1]
                project['training_mod_status_str'] = Database.projectStatusToStr( result[1] )
                project['segmentation_mod_status'] = result[2]
                project['segmentation_mod_status_str'] = Database.projectStatusToStr( result[2] )
                project['initial_model'] = result[3]
                project['model_type'] = result[4]
                project['sample_size'] = result[5]
                project['learning_rate'] = result[6]
                project['momentum'] = result[7]
                project['batch_size'] = result[8]
                project['epochs'] = result[9]
                project['train_time'] = result[10]
                project['sync_time'] = result[11]
                project['model_mod_time'] = result[12]
                project['locked'] = result[13]
	        projectId = project["id"]
                project['labels'] = Database.getLabels( projectId )
                project['hidden_layers'] = Database.getHiddenUnitss( projectId )
                project['images'] = Database.getTasks( projectId )
        return project


    @staticmethod
    def getProject(projectId):
        project = None
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Id, TrainingModuleStatus, PredictionModuleStatus, "
            cmd += "BaseModel, Type, PatchSize, LearningRate, "
            cmd += "Momentum, BatchSize, Epoch, TrainTime, SyncTime, ModelModifiedTime, Lock "
            cmd += "FROM Project WHERE Id='%s'"%(projectId)

            cur.execute( cmd )
            results = cur.fetchall()
            if len(results) > 0:
                result  = results[0]
                project = {}
                project["id"] = result[0]
                project['training_mod_status'] = result[1]
                project['training_mod_status_str'] = Database.projectStatusToStr( result[1] )
                project['segmentation_mod_status'] = result[2]
                project['segmentation_mod_status_str'] = Database.projectStatusToStr( result[2] )
                project['initial_model'] = result[3]
                project['model_type'] = result[4]
                project['sample_size'] = result[5]
                project['learning_rate'] = result[6]
                project['momentum'] = result[7]
                project['batch_size'] = result[8]
                project['epochs'] = result[9]
                project['train_time'] = result[10]
                project['sync_time'] = result[11]
                project['model_mod_time'] = result[12]
                project['locked'] = result[13]
                projectId = project["id"]
                project['labels'] = Database.getLabels( projectId )
                project['hidden_layers'] = Database.getHiddenUnitss( projectId )
                project['images'] = Database.getTasks( projectId )

		'''
                project["id"] = result[0]
                project["name"] = result[1]
                project['training_mod_status'] = result[2]
		project['training_mod_status_str'] = Database.projectStatusToStr( result[2] )
                project['segmentation_mod_status'] = result[3]
		project['segmentation_mod_status_str'] = Database.projectStatusToStr( result[3] )
                project['initial_model'] = result[4]
                project['model_type'] = result[5]
                project['sample_size'] = result[6]
                project['learning_rate'] = result[7]
                project['momentum'] = result[8]
                project['batch_size'] = result[9]
                project['epochs'] = result[10]
                project['train_time'] = result[11]
                project['sync_time'] = result[12]
		project['model_mod_time'] = result[13]
		project['locked'] = result[14]
                project['labels'] = Database.getLabels( projectId )
                project['hidden_layers'] = Database.getHiddenUnitss( projectId )
                project['images'] = Database.getTasks( projectId )
		'''
        return project


    @staticmethod
    def getModuleStatus(modName, pId):
	col = 'TrainingModuleStatus' if modName == 'training' else 'PredictionModuleStatus'
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT %s from Project WHERE Id='%s'"%(col, pId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
		return result[0]
	return 0

    @staticmethod
    def getProjectNames():
        names = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT * from Project "
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                names.append( result[0] )
        return names

    @staticmethod
    def getProjects():
        projects = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Id, TrainingModuleStatus, PredictionModuleStatus, "
            cmd += "BaseModel, Type, PatchSize, LearningRate, "
            cmd += "Momentum, BatchSize, Epoch, TrainTime, SyncTime, ModelModifiedTime, StartTime "
            cmd += "FROM Project ORDER BY StartTime DESC"
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                project = {}
                project["id"] = result[0]
                project['training_mod_status'] = result[1]
                project['segmentation_mod_status'] = result[2]
                project['initial_model'] = result[3]
                project['model_type'] = result[4]
                project['sample_size'] = result[5]
                project['learning_rate'] = result[6]
                project['momentum'] = result[7]
                project['batch_size'] = result[8]
                project['epochs'] = result[9]
                project['train_time'] = result[10]
                project['sync_time'] = result[11]
		project['start_time'] = result[12]
                project['labels'] = Database.getLabels( project["id"] )
                project['hidden_layers'] = Database.getHiddenUnitss( project["id"] )
                project['images'] = Database.getTasks( project["id"] )
                projects.append( project )
	return projects

    @staticmethod
    def finishSaveModel( projectId ):
	now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd = "UPDATE Project SET Lock=0, ModelModifiedTime=? WHERE Id=?"
            val = (now, projectId)
            cur.execute( cmd, val )
            connection.commit()

    @staticmethod
    def beginSaveModel( projectId ):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd = "UPDATE Project SET Lock=0, ModelModifiedTime=? WHERE Id=?"
            val = (now, projectId)
            cur.execute( cmd, val )
            connection.commit()

    @staticmethod
    def getModelModifiedTime(projectId):
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT ModelModifiedTime FROM Project WHERE Id='%s'"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
		return result[0]
	return None

    @staticmethod
    def storeImage( project, image ):
        segFile = '%s/%s.%s.seg'%(Paths.Segmentation, image, project)
        annFile = '%s/%s.%s.json'%(Paths.Labels, image, project)

        if not os.path.exists( segFile ):
             segFile = None
        if not os.path.exists( annFile ):
            annFile = None
        Database.storeTask( project, image, annFile, segFile )


    @staticmethod
    def removeImage( project, image ):
        segFile = '%s/%s.%s.seg'%(Paths.Segmentation, image, project)
        annFile = '%s/%s.%s.json'%(Paths.Labels, image, project)
        if os.path.exists( segFile ):
	    print 'removing ', segFile
            os.remove( segFile )
        if os.path.exists( annFile ):
	    print 'removing ', annFile
            os.remove( annFile )
        Database.removeTask( project, image)


    #---------------------------------------------------------------------------------------------
    # HiddenUnits table operations
    #---------------------------------------------------------------------------------------------
    @staticmethod
    def storeHiddenUnitsUnit(projectId, layerId, numUnits ):
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "INSERT OR REPLACE INTO HiddenUnits"
            cmd += "(ProjectId, Id, Units)"
            cmd += "VALUES (?,?,?)"
            vals = (projectId, layerId, numUnits)
            cur.execute( cmd, vals )

    @staticmethod
    def getHiddenUnits(projectId, layerId):
        layer = None
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT * from HiddenUnits "
            cmd += "WHERE ProjectId='%s' AND Id='%s'"%(projectId, layerId)
            cur.execute( cmd )
            results = cur.fetchall()
            if len(results) > 0:
                result = results[0]
                layer = {}
                layer['id'] = result[1]
                layer['units'] = result[2]
        return layer

    @staticmethod
    def getHiddenUnitss(projectId):
        layers = []
        connection = lite.connect( DATABASE_NAME )
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT * from HiddenUnits WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                layer = {}
                layer['id'] = result[1]
                layer['units'] = result[2]
                layers.append( layer )
        return layers

    @staticmethod
    def projectStatusToStr( status ):
	if status == 1:
		return 'Active'
	elif status == 2:
		return 'Pending Annotations'
	else:
		return 'Inactive'

#---------------------------------------------------------------------------
# Entry point to the main function of the program.
#---------------------------------------------------------------------------
if __name__ == '__main__':
    print 'Icon database (installation interface)'

    if len(sys.argv) > 1 and sys.argv[1] == 'install':

        # start in a blank slate
        Database.reset()

        # install the tables
        Database.initialize()

        # install the default model
        projectId = 'test'
	project = {}
        project['id'] = projectId
        project['initial_model'] = None
        project['model_type']='MLP'
        project['sample_size']=39
        project['learning_rate']=0.01
        project['momentum']=0.9
        project['batch_size']=10
        project['epochs']=0
        project['train_time']=15
        project['sync_time']=30

        Database.storeLabel(projectId, 0, 'background', 255,0,0)
        Database.storeLabel(projectId, 1, 'membrane', 0,255,0)
        #Database.addProject(project, project, '', 'MLP', 39, 0.01, 0.9, 20, 20, 15, 20, False)
	Database.addProject(project)
        Database.storeHiddenUnitsUnit(projectId, 0, 500)
        Database.storeHiddenUnitsUnit(projectId, 1, 500)
        Database.storeHiddenUnitsUnit(projectId, 2, 500)

	# setup training set
        imagePaths = glob.glob('%s/*.tif'%(Paths.TrainGrayscale))

	# setup the first 20 images as a training set
	i = 0
	for path in imagePaths:
		name = Utility.get_filename_noext( path )
		Database.storeImage(project['id'], name )
		print 'adding %s...'%(name)
		i += 1
		if i > 20:
			break
