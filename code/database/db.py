#-------------------------------------------------------------------------------------------
# db.py
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
#-------------------------------------------------------------------------------------------

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

from project import Project
from label import Label
from image import Image
from stats import TrainingStats

base_path = os.path.dirname(__file__)
sys.path.insert(2,os.path.join(base_path, '../common'))
DATABASE_NAME = os.path.join(base_path, '../../data/database/icon.db')

class DB:

    # Base SQL query for project
    QueryProject  = "SELECT Id, Type, Revision, BaseModel, "
    QueryProject += "TrainTime, SyncTime, Lock, "
    QueryProject += "PatchSize, BatchSize, "
    QueryProject += "LearningRate, Momentum, "
    QueryProject += "Epoch, TrainingModuleStatus, "
    QueryProject += "PredictionModuleStatus, CreationTime, "
    QueryProject += "StartTime, ModelModifiedTime, "
    QueryProject += "ActiveMode FROM Project "


    QueryImage    = "SELECT ImageId, Purpose, "
    QueryImage   += "SegmentationRequestTime, SegmentationTime, "
    QueryImage   += "SegmentationPriority, SegmentationFile, "
    QueryImage   += "TrainingTime, TrainingPriority, TrainingScore, "
    QueryImage   += "AnnotationTime, AnnotationLockTime, AnnotationStatus, "
    QueryImage   += "AnnotationLockId, AnnotationFile, "
    QueryImage   += "ModelModifiedTime, CreationTime, StartTime "
    QueryImage   += "FROM ImageView "

    PURPOSE_TRAIN_STR = 'train'
    PURPOSE_VALID_STR = 'valid'
    PURPOSE_NONE_STR = ''

    PURPOSE_TRAIN_INT = 0
    PURPOSE_VALID_INT = 1
    PURPOSE_NONE_INT = 2

    @staticmethod
    def addImage(projectId, imageIndex, purpose):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "INSERT INTO Image(ProjectId, ImageId, Purpose) "
            cmd += "VALUES(?,?,?)"
            vals = (projectId, imageIndex, DB.purpose_str_to_int(purpose))
            cur.execute( cmd, vals )

    #--------------------------------------------------------------------------------
    # BATCH
    #--------------------------------------------------------------------------------
    @staticmethod
    def storeTrainingStats(projectId, valLoss, avgCost, mode=0):

        #print 'id:', projectId, 'vl:', valLoss, 'ac:', avgCost, 'm:', mode
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "INSERT OR REPLACE INTO "
            cmd += "TrainingStats(ProjectId, ValidationError, "
            cmd += "TrainingCost, TrainingTime, ProjectMode ) "
            cmd += "VALUES(?,?,?,?,?)"
            vals = (projectId, valLoss, avgCost, now, mode)
            cur.execute( cmd, vals )

    # returns a list of live stats
    @staticmethod
    def getTrainingStats(projectId, projectMode=0):
        stats = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT ValidationError, TrainingCost, TrainingTime "
            cmd += "FROM TrainingStats WHERE ProjectId=? AND ProjectMode=?"
            cmd += "ORDER BY TrainingTime ASC LIMIT 100"
            vals = (projectId, projectMode)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            for result in results:
                stats.append( TrainingStats(
                        validationError = result[0],
                        trainingCost = result[1],
                        trainingTime = result[2]) )
        return stats

    @staticmethod
    def removeTrainingStats(projectId, projectMode=0):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "DELETE FROM TrainingStats WHERE "
            cmd += "ProjectId=? AND ProjectMode=?"
            vals = (projectId, projectMode)
            cur.execute( cmd, vals )


    #--------------------------------------------------------------------------------
    # PROJECTS
    #--------------------------------------------------------------------------------
    @staticmethod
    def toProject(result):
        if result == None:
            return None
        id                       = result[0]
        type                     = result[1]
        revision                 = result[2]
        project                  = Project(id, type, revision=revision)
        project.baseModel        = result[3]
        project.trainTime        = result[4]
        project.syncTime         = result[5]
        project.locked           = result[6]
        project.patchSize        = result[7]
        project.batchSize        = result[8]
        project.learningRate     = result[9]
        project.momentum         = result[10]
        project.epochs           = result[11]
        project.trainingStatus   = result[12]
        project.predictionStatus = result[13]
        project.creationTime     = result[14]
        project.startTime        = result[15]
        project.modelTime        = result[16]
        project.activemode       = result[17]
        project.path             = '%s/best_%s.%s.%d.pkl'%(Paths.Models, id, type, revision )
        project.path_offline     = '%s/best_%s.%s.offline.pkl'%(Paths.Models, id, type )
        project.path             = project.path.lower()
        project.path_offline     = project.path_offline.lower()
        project.labels           = DB.getLabels( id )
        project.nKernels         = DB.getNumKernels( id, type )
        project.kernelSizes      = DB.getKernelSizes( id, type )
        project.hiddenUnits      = DB.getHiddenUnits( id, type )
        project.images           = DB.getAllImages( id )
        #project.validation_images= DB.getImages( id, purpose=1 )
        project.offline          = DB.getOfflinePerformance( id )
        project.online           = DB.getOnlinePerformance( id )
        project.baseline         = DB.getBaselinePerformance( id )
        project.stats            = DB.getTrainingStats( id )
        return project

    @staticmethod
    def purpose_str_to_int(str):
        if str == DB.PURPOSE_VALID_STR:
            return DB.PURPOSE_VALID_INT
        elif str == DB.PURPOSE_TRAIN_STR:
            return DB.PURPOSE_TRAIN_INT
        else:
            return DB.PURPOSE_NONE_INT

    @staticmethod
    def purpose_int_to_str(i):
        if i == DB.PURPOSE_VALID_INT:
            return DB.PURPOSE_VALID_STR
        elif i == DB.PURPOSE_TRAIN_INT:
            return DB.PURPOSE_TRAIN_STR
        else:
            return DB.PURPOSE_NONE_STR

    @staticmethod
    def getProject(id):
        result = None
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = DB.QueryProject
            cmd += "WHERE Id='%s'"%(id)
            cur.execute( cmd )
            results = cur.fetchall()
            result  = results[0] if len(results) > 0 else None
        return DB.toProject( result )

    @staticmethod
    def getActiveProject():
        result = None
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = DB.QueryProject
            cmd += "WHERE TrainingModuleStatus > 0"
            cur.execute( cmd )
            results = cur.fetchall()
            result  = results[0] if len(results) > 0 else None
        return DB.toProject( result )

    @staticmethod
    def getModuleStatus(modName, pId):
        col = 'TrainingModuleStatus' if modName == 'training' else 'PredictionModuleStatus'
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT %s from Project WHERE Id='%s'"%(col, pId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                return result[0]
        return 0

    @staticmethod
    def getRevision(projectId):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Revision from Project WHERE Id='%s'"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                return result[0]
        return 0


    @staticmethod
    def getProjectNames():
        names = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT * from Project ORDER BY Id"
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                names.append( result[0] )
        return names

    @staticmethod
    def getProjects():
        projects = []
        results  = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = DB.QueryProject
            cmd += "ORDER BY StartTime DESC"
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                projects.append( DB.toProject( result ) )
        return projects

    @staticmethod
    def updateProject(project):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Project SET "
            cmd += "BatchSize=?, LearningRate=?, Momentum=? "
            cmd += "WHERE Id=?"
            vals = (project.batchSize,
                    project.learningRate,
                    project.momentum,
                    project.id)
            print 'values:'
            print vals
            cur.execute( cmd, vals )

    @staticmethod
    def storeProject(project):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "INSERT OR REPLACE INTO Project("
            cmd += "Id, Type, BaseModel, PatchSize, LearningRate, "
            cmd += "Momentum, BatchSize, Epoch, TrainTime, SyncTime, "
            cmd += "TrainingModuleStatus, PredictionModuleStatus, ModelModifiedTime, "
            cmd += "CreationTime )"
            cmd += "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            vals = (project.id,
                    project.type,
                    project.baseModel,
                    project.patchSize,
                    project.learningRate,
                    project.momentum,
                    project.batchSize,
                    project.epochs,
                    project.trainTime,
                    project.syncTime,
                    0,
                    0,
                    now,
                    now)
            cur.execute( cmd, vals )

        # print 'db.storeProject:'
        # print project.id
        # print project.type
        # print 'hiden:',project.hiddenUnits
        # print 'kernsizes:', project.kernelSizes
        # print 'nkernels:',project.nKernels

        DB.storeHiddenUnits( project.id, project.type, project.hiddenUnits )
        DB.storeKernels( project.id, project.type, project.nKernels, project.kernelSizes )

        for label in project.labels:
            DB.storeLabel( project.id, project.type, label )

        for image in project.images:
            DB.storeImage( project.id, project.type, image )

    @staticmethod
    def finishSaveModel( projectId, revision ):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd = "UPDATE Project SET Lock=0, Revision=?, ModelModifiedTime=? WHERE Id=?"
            val = (revision, now, projectId)
            cur.execute( cmd, val )

    @staticmethod
    def beginSaveModel( projectId ):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd = "UPDATE Project SET Lock=0, ModelModifiedTime=? WHERE Id=?"
            val = (now, projectId)
            cur.execute( cmd, val )

    @staticmethod
    def getModelModifiedTime(projectId):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT ModelModifiedTime FROM Project WHERE Id='%s'"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                return result[0]
        return None

    @staticmethod
    def removeProject(projectId):
        print 'DB.removeproject..:', projectId
        project = None
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "DELETE from Project WHERE Id='%s'"%(projectId)
            cur.execute( cmd )

            cmd  = "DELETE from Image WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )

            cmd = "DELETE FROM HiddenUnits WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )

            cmd = "DELETE FROM NumKernels WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )

            cmd = "DELETE FROM KernelSize WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )

            cmd = "DELETE FROM TrainingStats WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )

            cmd = "DELETE FROM Label WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )

            cmd = "DELETE FROM Performance WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )

    @staticmethod
    def stopProject(projectId):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Project SET TrainingModuleStatus=0, "
            cmd += "PredictionModuleStatus=0 WHERE Id='%s'"%(projectId)
            cur.execute( cmd )

    @staticmethod
    def startProject(projectId, activeMode=0):
        print 'activemode:', activeMode
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Project SET TrainingModuleStatus=0, "
            cmd += "PredictionModuleStatus=0, ActiveMode=0"
            # WHERE Id='%s'"%(projectId)
            cur.execute( cmd )

            cmd  = "UPDATE Project SET TrainingModuleStatus=1, PredictionModuleStatus=1, "
            cmd += "StartTime=?, ActiveMode=? WHERE Id=?"
            vals = (now, activeMode, projectId)
            cur.execute( cmd, vals )

    @staticmethod
    def isStopped(projectId):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT TrainingModuleStatus FROM Project WHERE Id='%s'"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            if len(results) > 0:
                return (results[0][0] == 1)
        return False

    @staticmethod
    def updatePredictionModuleStatus(projectId, status):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Project SET PredictionModuleStatus=? WHERE Id=?"
            vals = (status, projectId)
            cur.execute( cmd, vals )

    @staticmethod
    def updateTrainingStatus(projectId, status):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Project SET TrainingModuleStatus=? "
            cmd += "WHERE Id=?"
            vals = (status, projectId)
            cur.execute( cmd, vals )

    @staticmethod
    def finishLoadingTrainingset(projectId):
        DB.updateTrainingStatus( projectId, 1)
        DB.updateTrainingImagesLoaded( projectId )



    #--------------------------------------------------------------------------------
    # IMAGES
    #--------------------------------------------------------------------------------
    @staticmethod
    def storeTrainingScore(projectId, imageId, score):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Image SET TrainingScore=? "
            cmd += "WHERE ProjectId=? AND ImageId=?"
            vals = (score, projectId, imageId)
            cur.execute( cmd, vals )

    @staticmethod
    def toImage( result ):
        if result == None:
            return

        image                      = Image( result[0],purpose=result[1] )
        image.segmentationReqTime  = result[2]
        image.segmentationTime     = result[3]
        image.segmentationPriority = result[4]
        image.segmentationFile     = result[5]

        image.trainingTime         = result[6]
        image.trainingPriority     = result[7]
        image.trainingScore        = result[8]

        image.annotationTime       = result[9]
        image.annotationLockTime   = result[10]
        image.annotationStatus     = result[11]
        image.annotationLockId     = result[12]
        image.annotationFile       = result[13]

        image.modelModifiedTime    = result[14]
        image.creationTime         = result[15]
        image.startTime            = result[16]


        # set a flag when new segmentation is available
        creationTime = time.strptime(image.creationTime, '%Y-%m-%d %H:%M:%S')
        startTime = time.strptime(image.startTime, '%Y-%m-%d %H:%M:%S')
        segTime = time.strptime(image.segmentationTime, '%Y-%m-%d %H:%M:%S')
        modelTime = time.strptime(image.modelModifiedTime, '%Y-%m-%d %H:%M:%S')
        image.hasNewModel = segTime < modelTime and startTime > creationTime
        return image


    @staticmethod
    def getAllImages(projectId):
        images = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = DB.QueryImage
            cmd += "WHERE ProjectId='%s'"%(projectId)
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                images.append( DB.toImage( result ) )
        return images
       
    # purpose (0=training,1=validation,2=test,3=all,4=annotated)
    @staticmethod
    def getImages(
        projectId,
        purpose=3,
        annotated=False,
        new=False,
        trainingNew=False,
        segmentation=False):
        #purpose_str = DB.purpose_int_to_str(purpose)
        images = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = DB.QueryImage
            #cmd  = "SELECT ImageId, AnnotationFile, "
            #cmd += "SegmentationFile, TrainingScore "
            cmd += "WHERE ProjectId=? AND Purpose=? "
            if annotated:
                cmd += "AND AnnotationFile is NOT NULL "
                if new:
                    cmd += "AND TrainingTime < AnnotationTime "
                    cmd += "ORDER BY AnnotationTime DESC "

            if purpose == 3:
                cmd += "ORDER BY TrainingScore DESC"

            #print 'projectId:',projectId, 'purpose:', purpose
            #print 'cmd:', cmd
            vals = (projectId, purpose)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            for result in results:
                images.append( DB.toImage( result ) )

        return images

    @staticmethod
    def getTrainingImages( projectId, new=False ):
        imgs = DB.getImages( projectId, purpose=0, annotated=True, new=new )
        return imgs

    @staticmethod
    def getImage(projectId, imageId):
        image = Image( imageId )
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = DB.QueryImage
            cmd += "WHERE ProjectId=? AND ImageId=?"
            vals = (projectId, imageId)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            if len(results) > 0:
                image = DB.toImage( results[0] )
        return image

    @staticmethod
    def storeImage(projectId, projectType, image):
        now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT * FROM Image WHERE ProjectId=? AND ProjectType=? AND ImageId=?"
            vals = (projectId, projectType, image.id)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            if len(results) > 0:
                return
            else:
                    cmd  = "INSERT INTO Image(Purpose, AnnotationFile, SegmentationFile, "
                    cmd += "ProjectId, ProjectType, ImageId, SegmentationTime, TrainingTime, "
                    cmd += "AnnotationTime, SegmentationRequestTime) "
                    cmd += "VALUES(?,?,?,?,?,?,?,?,?,?)"
            vals = (image.purpose, 
                    image.annotationFile,
                    image.segmentationFile,
                    projectId,
                    projectType,
                    image.id,
                    now, now, now, now)
            cur.execute( cmd, vals )

    @staticmethod
    def removeImage( projectId, imageId ):
        # remove the image files
        segFile = '%s/%s.%s.seg'%(Paths.Segmentation, imageId, projectId)
        annFile = '%s/%s.%s.json'%(Paths.Labels, imageId, projectId)
        if os.path.exists( segFile ):
            os.remove( segFile )
        if os.path.exists( annFile ):
            os.remove( annFile )

        # remove the database record
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "DELETE FROM Image WHERE ProjectId=? AND ImageId=?"
            vals = (projectId, imageId)
            cur.execute( cmd, vals )

    @staticmethod
    def getImageNames(projectId, purpose=0):
        names = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT ImageId FROM Image WHERE ProjectId=? AND Purpose=?"
            vals = (projectId, purpose)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            for result in results:
                names.append(result[0])
        return names

    @staticmethod
    def updateTrainingImagesLoaded( projectId):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "UPDATE Image SET TrainingPriority = 0, TrainingTime=? "
            cmd += "WHERE ProjectId=?"
            vals = (now, projectId)
            cur.execute( cmd, vals )


    @staticmethod
    def finishPrediction( projectId, imageId, duration, modelModTime ):
        print 'DB.finishPrediction...'
        print 'id:',projectId
        print 'img:', imageId
        print 'dur:', duration
        print 'mod:', modelModTime

        segTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        segFile = '%s/%s.%s.seg'%(Paths.Segmentation, imageId, projectId)
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd = "UPDATE Image SET SegmentationTime=?, SegmentationFile=?, "
            cmd += "SegmentationPriority=0 WHERE ProjectId=? AND ImageId=?"
            val = (segTime, segFile, projectId, imageId)
            cur.execute( cmd, val )

            cmd  = "UPDATE Project SET SyncTime=? WHERE Id=?"
            vals = (duration, projectId)
            cur.execute( cmd, vals )

    @staticmethod
    def requestSegmentation(projectId, imageId):
        print 'db.requestSegmentation...', projectId, imageId
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Image SET SegmentationPriority=1, SegmentationRequestTime=? "
            cmd += "WHERE ProjectId=? AND ImageId=?"
            vals = (now, projectId, imageId)
            cur.execute( cmd, vals )

    @staticmethod
    def saveAnnotations(projectId, imageId, annFile):
        print 'annfile: ', annFile
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "UPDATE Image SET TrainingPriority=1, "
            cmd += "SegmentationPriority=1, AnnotationFile=?, AnnotationTime=? "
            cmd += "WHERE ProjectId=? AND ImageId=?"
            vals = (annFile, now, projectId, imageId)
            cur.execute( cmd, vals )


    @staticmethod
    def getPredictionImages(projectId, priority=0):
        images = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = DB.QueryImage
            #cmd += "WHERE Lock == 0 AND SegmentationTime < ModelModifiedTime "
            #cmd += "WHERE Lock == 0 AND SegmentationTime != ModelModifiedTime "
            #cmd += "WHERE SegmentationTime != ModelModifiedTime "
            cmd += "WHERE ProjectId=? AND SegmentationPriority=? ORDER BY SegmentationRequestTime DESC"
            #cmd += "WHERE ProjectId=? AND SegmentationPriority=? ORDER BY SegmentationRequestTime DESC"
            vals = (projectId, priority)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            for result in results:
                image = DB.toImage( result )
                images.append ( image )
        return images


    @staticmethod
    def lockImage(projectId, imageId):
        guid = uuid.uuid1().hex
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "UPDATE Image SET AnnotationStatus=1, AnnotationLockId=?, "
            cmd += "AnnotationLockTime=? WHERE ProjectId=? AND ImageId=?"
            vals = (guid, now, projectId, imageId)
            cur.execute( cmd, vals )
        return guid

    @staticmethod
    def refreshAnnotationLock(projectId, imageId, guid):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "UPDATE Image SET AnnotationLockTime=? "
            cmd += "WHERE AnnotationLockId=? AND ProjectId=? AND ImageId=?"
            vals = (now, guid, projectId, imageId)
            cur.execute( cmd, vals )
        return guid

    #--------------------------------------------------------------------------------
    # HIDDEN UNITS
    #--------------------------------------------------------------------------------
    @staticmethod
    def getHiddenUnits(projectId, projectType):
        units = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Units FROM HiddenUnits "
            cmd += "WHERE ProjectId=? AND ProjectType=? "
            cmd += "ORDER BY Id"
            vals = (projectId, projectType)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            for result in results:
                units.append( result[0] )
        return units

    @staticmethod
    def storeHiddenUnits(projectId, projectType, units):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            for index, unit in enumerate(units):
                cmd  = "INSERT OR REPLACE INTO HiddenUnits"
                cmd += "(ProjectId, ProjectType, Id, Units)"
                cmd += "VALUES (?,?,?,?)"
                vals = (projectId, projectType, index , unit)
                cur.execute( cmd, vals )


    @staticmethod
    def removeHiddenUnits(projectId):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "DELETE FROM HiddenUnits WHERE "
            cmd += "ProjectId='%s'"%(projectId)
            cur.execute( cmd )

    #--------------------------------------------------------------------------------
    # KERNELS
    #--------------------------------------------------------------------------------
    @staticmethod
    def getNumKernels(projectId, projectType):
        data = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Count FROM NumKernels "
            cmd += "WHERE ProjectId=? AND ProjectType=? "
            cmd += "ORDER BY Id"
            vals = (projectId, projectType)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            for result in results:
                data.append( result[0] )
        return data


    @staticmethod
    def getKernelSizes(projectId, projectType):
        data = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Size FROM KernelSize "
            cmd += "WHERE ProjectId=? AND ProjectType=? "
            cmd += "ORDER BY Id"
            vals = (projectId, projectType)
            cur.execute( cmd, vals )
            results = cur.fetchall()
            for result in results:
                data.append( result[0] )
        return data

    @staticmethod
    def removeKernelSizes(projectId):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "DELETE FROM KernelSize WHERE "
            cmd += "ProjectId='%s'"%(projectId)
            cur.execute( cmd )


    @staticmethod
    def storeKernels(projectId, projectType, nKernels, kernelSizes):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            for index, value in enumerate(nKernels):
                cmd  = "INSERT OR REPLACE INTO NumKernels"
                cmd += "(ProjectId, ProjectType, Id, Count)"
                cmd += "VALUES (?,?,?,?)"
                vals = (projectId, projectType, index , value)
                cur.execute( cmd, vals )

            for index, value in enumerate(kernelSizes):
                cmd  = "INSERT OR REPLACE INTO KernelSize"
                cmd += "(ProjectId, ProjectType, Id, Size)"
                cmd += "VALUES (?,?,?,?)"
                vals = (projectId, projectType, index , value)
                cur.execute( cmd, vals )


    @staticmethod
    def removeNumKernels(projectId):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "DELETE FROM NumKernels WHERE "
            cmd += "ProjectId='%s'"%(projectId)
            cur.execute( cmd )


    #--------------------------------------------------------------------------------
    # LABELS
    #--------------------------------------------------------------------------------
    @staticmethod
    def getLabels(projectId):
        labels = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "SELECT Id, Name, R, G, B "
            cmd += "FROM Label WHERE ProjectId='%s' "%(projectId)
            cmd += "ORDER BY Id"
            cur.execute( cmd )
            results = cur.fetchall()
            for result in results:
                label = Label(  index=result[0],
                                name=result[1],
                                r=result[2],
                                g=result[3],
                                b=result[4])
                labels.append( label )
        return labels


    @staticmethod
    def storeLabel(projectId, projectType, label):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "INSERT OR REPLACE INTO Label( "
            cmd += "ProjectId, ProjectType, Id, Name, R, G, B)"
            cmd += "VALUES (?,?,?,?,?,?,?)"
            vals = (projectId, projectType, label.index, label.name, label.r, label.g, label.b)
            cur.execute( cmd, vals )


    @staticmethod
    def removeLabels(projectId):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "DELETE FROM Label WHERE "
            cmd += "ProjectId='%s'"%(projectId)
            cur.execute( cmd )


    #--------------------------------------------------------------------------------
    # PERFORMANCE
    #--------------------------------------------------------------------------------
    @staticmethod
    def storePerformance(projectId, type, threshold, variationInfo, pixelError):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        connection   = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "INSERT OR REPLACE INTO "
            cmd += "Performance(ProjectId, Type, Threshold,"
            cmd += "VariationInfo, PixelError, CreationTime)"
            cmd += "VALUES(?,?,?,?,?,?)"
            vals = (projectId, type, threshold, variationInfo, pixelError, now)
            cur.execute( cmd, vals )

    @staticmethod
    def removePerformances(projectId, type):
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur  = connection.cursor()
            cmd  = "DELETE FROM Performances WHERE "
            cmd += "ProjectId=? AND Type=?"
            vals = (projectId, type)
            cur.execute( cmd )

    @staticmethod
    def getPerformance(projectId, type):
        perf = []
        connection = lite.connect( DATABASE_NAME, timeout=30)
        with connection:
            cur = connection.cursor()
            cmd  = "SELECT Threshold, VariationInfo, PixelError, CreationTime FROM Performance "
            cmd += "WHERE ProjectId=? AND Type=?"
            vals = (projectId, type)
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
    def getOfflinePerformance(projectId):
        return DB.getPerformance(projectId, 'offline')

    @staticmethod
    def getOnlinePerformance(projectId):
        return DB.getPerformance(projectId, 'online')

    @staticmethod
    def getBaselinePerformance(projectId):
        return DB.getPerformance(projectId, 'baseline')

    # MISC
    @staticmethod
    def projectStatusToStr( status ):
        if status == 1:
            return 'Active'
        elif status == 2:
            return 'Pending Annotations'
        else:
            return 'Inactive'
