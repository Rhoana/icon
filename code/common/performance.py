#-bels--------------------------------------------------------------------------
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
import numpy as np
from PIL import Image
import mahotas
import glob
import partition_comparison
import StringIO
import csv
from math import log


base_path = os.path.dirname(__file__)
#sys.path.insert(1,os.path.join(base_path, '../common'))
#sys.path.insert(1,os.path.join(base_path, '../offline'))

from paths import Paths
from utility import *
from database import Database

def variation_of_information(X, Y):
  n = float(sum([len(x) for x in X]))
  sigma = 0.0
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n
      if r > 0.0:
        sigma += r * (log(r / p, 2) + log(r / q, 2))
  return abs(sigma), abs(sigma) / log(n, 2)


#---------------------------------------------------------------------------
# Thresholds used for computing performance
#---------------------------------------------------------------------------
class Performance:

    # thresholds for performance evaluation
    #Thresholds = np.arange(0.001, 1.0, 0.0005)
    Thresholds = np.arange(0.0, 1.001, 0.001)
    #Thresholds = np.arange(0.0, 1.1, 0.1)

    #-------------------------------------------------------------------
    # constructs a performance object
    #-------------------------------------------------------------------
    def __init__(self, image_id):
        self.image_id        = image_id
        self.results         = []
        '''
        self.pixel_errors    = []
        self.variation_infos = []
        '''

    def get(self, threshold_index ):
        return self.results[ threshold_index ]

    #-------------------------------------------------------------------
    # retrieves the pixel error for the specified threshold index
    #-------------------------------------------------------------------
    def get_pixel_error( self, threshold_index ):
        return self.pixel_errors[threshold_index]

    #-------------------------------------------------------------------
    # retrieves the variation of information for the sepcified
    # threshold index
    #-------------------------------------------------------------------
    def get_variation_info( self, threshold_index ):
        return self.variation_infos[threshold_index]


    def voi(self, prob, label):
        prob = np.int64(1-prob)
        f_image, n_labels  = mahotas.label(prob)
        f_image = np.int64( f_image.flatten() )
        f_label = np.int64( label.flatten() )
        return n_labels, partition_comparison.variation_of_information( f_image, f_label )

    def nperror(self, prob, image):
        image_norm = Utility.normalizeImage( image )

        # number of ground truth pixels in each label
        n_pixels_b = np.sum( image_norm == 0 )
        n_pixels_m = np.sum( image_norm == 1 )
        n_pixels   = image_norm.shape[0]*image_norm.shape[1]

        n_probs_b  = len(prob[(image_norm == prob) & (prob == 0)])
        n_probs_m  = len(prob[(image_norm == prob) & (prob == 1)])

        accuracy_m = 0.0 if n_probs_m == 1 else float(n_probs_m)/n_pixels_m
        accuracy_b = 0.0 if n_probs_b == 0 else float(n_probs_b)/n_pixels_b

        #print 'nb:',n_pixels_b, 'npb:', n_probs_b, 'nm:', n_pixels_m, 'npm:', n_probs_m,   accuracy_b, accuracy_m

        diff = np.absolute( image_norm - prob )
        sum  = np.sum( diff )
        accuracy = 0.0
        if sum > 0:
            accuracy = float(sum)/(diff.shape[0]*diff.shape[1])

        sum = n_probs_b + n_probs_m
        if sum > 0:
            accuracy = float(sum)/n_pixels
        #accuracy = 1.0 - accuracy
        return accuracy, accuracy_b, accuracy_m

    def perror(self, prob, image):
        image_norm = Utility.normalizeImage( image )

        # number of ground truth pixels in each label
        n_pixels_label0 = np.sum( image_norm == 0 )
        n_pixels_label1 = np.sum( image_norm == 1 )
        n_pixels        = n_pixels_label0 + n_pixels_label1
      
 
        # actual number correctly classifier pixels in each label
        n_pixels_correct_label0 = len(prob[(image_norm == prob) & (prob == 0)])
        n_pixels_correct_label1 = len(prob[(image_norm == prob) & (prob == 1)])  
        n_pixels_correct        = n_pixels_correct_label0 + n_pixels_correct_label1
        
        # error for each label with respect to correctly classified pixels
        pixel_error_label0 = 0.0
        pixel_error_label1 = 0.0
        pixel_error = 0.0
        if n_pixels_correct_label0 > 0:
            pixel_error_label0 = 1.0 - (float(n_pixels_correct_label0)/n_pixels_label0)
        if n_pixels_correct_label1 > 0:
            pixel_error_label1 = 1.0 - (float(n_pixels_correct_label1)/n_pixels_label1)
        if n_pixels_correct > 0:
            pixel_error = 1.0 - (float(n_pixels_correct)/n_pixels)

        '''
        print 'pmin:', np.min(prob)
        print 'pmax:', np.max(prob)
        print 'mmin:', np.min(image_norm)
        print 'mmax:', np.max(image_norm)
        print 'n_pixels_label0:', n_pixels_label0
        print 'n_pixels_label1:', n_pixels_label1
        print 'n_pixels_correct_label0:', n_pixels_correct_label0
        print 'n_pixels_correct_label1:', n_pixels_correct_label1
        print 'pixel_error_label0:', pixel_error_label0
        print 'pixel_error_label1:', pixel_error_label1
        '''
        return pixel_error, pixel_error_label0, pixel_error_label1

    #-------------------------------------------------------------------
    # compute performance results of a model given its 2-dimensional
    # array of probabilities and an image name
    #-------------------------------------------------------------------
    def compute(self, prob, model=None):
        postfix  = self.image_id.replace('input', 'labels')
        labelPath = '%s/%s.tif'%(Paths.TestLabels, postfix)
        memPath = '%s/%s.tif'%(Paths.Membranes, postfix)
        #labelPath = '/n/home00/fgonda/icon/data/reference/labels/test/%s.tif'%(postfix)


        print 'labelPath:', labelPath
        print 'memPath:', memPath

        # load the membrane and label images
        mem_image = mahotas.imread( memPath )
        #label_image = mahotas.imread( labelPath )
        label_image = Image.open( labelPath )
        label_image = np.array( label_image.getdata() )
        print 'label_image:', label_image.shape
        print 'label_image type:', type(label_image)
        print 'mem_image:', mem_image.shape
        

        # flatten the labeled image since variation of information
        # expects a one-dimensional array
        #label_image = np.int64( label_image.flatten() )
        #mem_image = mem_image.flatten()

        # compute pixel error and variation of information
        # for each threshold value.
        for threshold in Performance.Thresholds:
            # make a copy of the probability map
            p = np.copy( prob )
            p = model.threshold( p, factor=threshold)
            p = np.int64( p )

            vi = self.voi(p, label_image)
            pe = self.perror(p, mem_image)


            self.results.append( (vi[0], vi[1], pe[0], pe[1], pe[2]) )

            '''
            #--------------------------------------------------------
            # compute the corresponding pixel error
            #--------------------------------------------------------
            p_flat = p.flatten()
            matched = (mem_image == p_flat)
            pixel_error = (1.0 - float( np.sum(matched) )/mem_image.shape[0])
            self.pixel_errors.append( pixel_error )

            #--------------------------------------------------------
                # compute the corresponding variation of information
            #--------------------------------------------------------
            # invert probabilities
            p_inverted = (1 - p)*255

            # label the probabilities
            p_labeled, n_labels = mahotas.label( p_inverted )
            p_labeled = np.int64(p_labeled)
            p_labeled = p_labeled.flatten()

            #print 'p_labeled:', len(p_labeled),  'label_image:', len(label_image), '#labels:', n_labels, 'threshold:', threshold
            # compute the variation of information
            vi = partition_comparison.variation_of_information( p_labeled, label_image )
            self.variation_infos.append( vi )
            '''


    #-------------------------------------------------------------------
    # compute performance results of a model given its 2-dimensional
    # array of probabilities and an image name
    #-------------------------------------------------------------------
    @staticmethod
    def measureOnline(model, projectId, mean, std, maxNumTests=1):
        Performance.measure( model, projectId, 'online', mean, std,maxNumTests=maxNumTests)

    #-------------------------------------------------------------------
    # compute performance results of a model given its 2-dimensional
    # array of probabilities and an image name
    #-------------------------------------------------------------------
    @staticmethod
    def measureOffline(model, projectId, mean, std, maxNumTests=1):
        Performance.measure( model, projectId, 'offline', mean, std,maxNumTests=maxNumTests)


    #-------------------------------------------------------------------
    # compute performance results of a model given its 2-dimensional
    # array of probabilities and an image name
    #-------------------------------------------------------------------
    @staticmethod
    def measure(model, projectId, perfType, mean, std, maxNumTests=1):

        # get the test set
        testImagePaths = glob.glob('%s/*.tif'%(Paths.TestGrayscale))

        # track the performance results
        performances = []

        i = 0

        # measure performance for each test image
        for path in testImagePaths:

            i += 1
            if i > maxNumTests:
                break

            # extract the name of the image from the path
            name = Utility.get_filename_noext( path )

            # load the test image
            print 'generating performance for...%s'%(name)
            test_image = mahotas.imread( path )
            test_image = Utility.normalizeImage( test_image )
            #success, test_image = Utility.get_image_padded(path, model.get_patch_size())

            print model
            # compute the probabilities of the test image
            prob = model.classify(image=test_image, mean=mean, std=std)

            #felix additions
            patchSize=model.patchSize
            border_patch = int(np.ceil(patchSize/2))
            border = int(np.ceil(np.sqrt(2*(border_patch**2))))
            '''
            prob[:border,:] = 0 #top
            prob[-border:,:] = 0 #bottom
            prob[:,:border] = 0 #left
            prob[:,-border:] = 0 #right
            '''

            output = StringIO.StringIO()
            output.write(prob.tolist())
            content = output.getvalue()

            name = Utility.get_filename_noext( path )
            p = '%s/%s.%s.%s.prob'%(Paths.Results,name,projectId, perfType)
            with open(p, 'w') as outfile:
                outfile.write(content)

            if True:
                continue

            # compute the pixel error and variation of information
            performance = Performance( image_id=name )
            performance.compute( prob, model )
            performances.append( performance )


        if True:
            return

        # save the performances
        Performance.save( projectId, perfType, performances )

    @staticmethod
    def measureGroundTruth(model, projectId, mean=0.5, std=1.0, maxNumTests=1):
        
        # get the test set
        test_images = glob.glob('%s/*.tif'%(Paths.Membranes))
        if len(test_images) == 0:
            return

        # extract the imageid from the file path

        performances = []

        i = 0
        for path in test_images:
            name = Utility.get_filename_noext( path )

            i += 1
            if i > maxNumTests:
                break

            print 'measuring...%s'%(path)

            # load the grayscale image and use it as probability
            prob = mahotas.imread( path )
            prob = Utility.normalizeImage( prob )
            '''
            prob_mean = mean
            prob = prob - np.tile(prob_mean,(np.shape(prob)[0],1))

            prob_std = std
            prob_std = np.clip(prob_std, 0.00001, np.max(prob_std))
            prob = prob / np.tile(prob_std,(np.shape(prob)[0],1))
            '''

            # measure the performance
            performance = Performance(  image_id=name )
            performance.compute( prob, model )
            performances.append( performance )

        # save the perofrmance
        Performance.save( projectId, 'groundtruth', performances )


    #-------------------------------------------------------------------
    # measure baseline performance the specified model path and save
    # results to central database
    #-------------------------------------------------------------------
    @staticmethod
    def measureBaseline(model, projectId, mean=0.5, std=1.0, maxNumTests=1):

        # get the test set
        test_images = glob.glob('%s/*.tif'%(Paths.TestGrayscale))
        if len(test_images) == 0:
            return

        # extract the imageid from the file path

        performances = []

        i = 0
        for path in test_images:
            name = Utility.get_filename_noext( path )

            i += 1
            if i > maxNumTests:
                break

            print 'measuring...%s'%(path)

            # load the grayscale image and use it as probability
            prob = mahotas.imread( path )
            prob = Utility.normalizeImage( prob )
            '''
            prob_mean = mean
            prob = prob - np.tile(prob_mean,(np.shape(prob)[0],1))

            prob_std = std
            prob_std = np.clip(prob_std, 0.00001, np.max(prob_std))
            prob = prob / np.tile(prob_std,(np.shape(prob)[0],1))
            '''

            # measure the performance
            performance = Performance(  image_id=name )
            performance.compute( prob, model )
            performances.append( performance )

        # save the perofrmance
        Performance.save( projectId, 'baseline', performances )


    #-------------------------------------------------------------------
    # measure baseline performance the specified model path
    #-------------------------------------------------------------------
    @staticmethod
    def save(project_id, perf_type, performances=[]):

        n_results = len(performances)
        n_thresholds = len(Performance.Thresholds)

        data = 'threshold,vi,pe\n\r'

        path = '%s/%s.%s.csv'%(Paths.Results, project_id, perf_type)
        with open(path, 'w') as csvfile:

            fieldnames = [  'threshold',
                            'variation_labels', 
                            'var_info',
                            'pixel_error', 
                            'pixel_error_bg', 
                            'pixel_error_mem']
            writer = csv.writer(csvfile)
            writer.writerow(fieldnames)

            # compute the average pixel error and variation of information
            # for each threshold and store the results in the SQL Lite database
            for threshold_index in range(n_thresholds):
                vi = 0.0
                pe = 0.0
                pe_membrane = 0.0
                pe_background = 0.0
                nlabels = 0

                # sum the pixel errors and variation infos
                for performance in performances:
                    p = performance.get( threshold_index )
                    #pe += performance.get_pixel_error( threshold_index )
                    #vi += performance.get_variation_info( threshold_index )
                    nlabels       += p[0]
                    vi            += p[1]
                    pe            += p[2]
                    pe_background += p[3]
                    pe_membrane   += p[4]

                # compute the averages
                vi /= n_results
                pe /= n_results
                pe_membrane /= n_results
                pe_background /= n_results
                nlabels /= n_results

                data = ['%.5f'%( Performance.Thresholds[ threshold_index ] ),
                        '%d'%(nlabels),
                        '%.5f'%( vi ),
                        '%.5f'%( pe ),
                        '%.5f'%( pe_background ),
                        '%.5f'%( pe_membrane )
                       ]
                writer.writerow( data )

                if True:
                    #print 'th:', Performance.Thresholds[ threshold_index ], 'vi:', vi, 'pe:', pe
                    continue

                '''
                # store results in the database
                Database.storeModelPerformance(
                        project_id,
                        perf_type,
                        Performance.Thresholds[ threshold_index ],
                        vi,
                        pe)
                #print 'th:', Performance.Thresholds[ threshold_index ], 'vi:', vi, 'pe:', pe
                '''


#---------------------------------------------------------------------------
# Entry point to the main function of the program.
#---------------------------------------------------------------------------
if __name__ == '__main__':
    print 'creating baseline performance project test'

    #if len(sys.argv) > 1 and sys.argv[1] == 'install':
    Performance.measureBaseline('testmlp', maxNumTests=2)
    
