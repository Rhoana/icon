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
import zlib
import StringIO
import base64
import time
import numpy as np
import glob
import json
import pickle
import shutil
import resource
import mahotas
import tifffile as tiff

from datetime import datetime
from paths import Paths

def enum(**enums):
    return type('Enum', (), enums)

class Utility:

    @staticmethod
    def pad_image(img, patchSize):
        mode   = 'symmetric'
        pad = patchSize
        img = np.pad(img, ((pad, pad), (pad, pad)), mode)
        return img

    #-------------------------------------------------------------------
    # extracts the name of the file without extension from a path
    #-------------------------------------------------------------------
    @staticmethod
    def get_filename_noext(path):
        basepath, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)
        return name

    @staticmethod
    def normalizeImage(img, saturation_level=0.05): #was 0.005
        sortedValues = np.sort( img.ravel())
        n_sortedvals = len(sortedValues)
        n_sortedmin = int(n_sortedvals * (saturation_level / 2))
        n_sortedmax = int(n_sortedvals * (1 - saturation_level / 2))
        minVal = np.float32(sortedValues[n_sortedmin]) #len(sortedValues) * (saturation_level / 2)])
        maxVal = np.float32(sortedValues[n_sortedmax]) #len(sortedValues) * (1 - saturation_level / 2)])
        normImg = np.float32(img - minVal) * (255 / (maxVal-minVal))
        normImg[normImg<0] = 0
        normImg[normImg>255] = 255
        return (np.float32(normImg) / 255.0)

    @staticmethod
    def oldnormalizeImage(img, saturation_level=0.05): #was 0.005
        sortedValues = np.sort( img.ravel())
        minVal = np.float32(sortedValues[len(sortedValues) * (saturation_level / 2)])
        maxVal = np.float32(sortedValues[len(sortedValues) * (1 - saturation_level / 2)])
        normImg = np.float32(img - minVal) * (255 / (maxVal-minVal))
        normImg[normImg<0] = 0
        normImg[normImg>255] = 255
        return (np.float32(normImg) / 255.0)


    @staticmethod
    def getImages( path ):
        images = []
        tifffiles = glob.glob(path + '/*.tif')
        for image in tifffiles:
            tokens = image.split('/');
            name = tokens[ len(tokens)-1 ]
            tokens = name.split('.')
            name = tokens[0]

            img = {}
            img['ann_file'] = ''
            img['image_id'] = name
            img['project_id'] = ''
            img['score'] = 0.0
            img['seg_file'] = ''
            images.append( img )
        return images


    @staticmethod
    def report_memused():
        mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        mem_mb = mem_kb/1000.0
        mem_gb = mem_kb/1000000.0
        Utility.report_status('mem used', '(%0.2f KB) - (%0.2f MB) - (%0.2f GB)'%(mem_kb, mem_mb, mem_gb))

    @staticmethod
    def saveJson(path, data):
        with open(path, 'w') as outfile:
            outfile.write( json.dumps(data) )

    @staticmethod
    def readJson(path):
        if os.path.exists( path ):
            with open(path) as json_file:
                return json.load( json_file )
        return None

    @staticmethod
    def get_patch(image, row, rowrange, sample_size, data_mean=None, data_std=None):
        #patch = image[row:row+numrows].flatten()
        #def get_test_data( img, sample_size ):
        nx, ny = image.shape
        nx -= sample_size*2
        ny -= sample_size*2
        #print 'nx:', nx, 'ny:', ny, 'rr:', rowrange
        size = sample_size
        h    = size/2
        x    = np.zeros((nx*rowrange, size*size), dtype=np.float32)
        #print image.shape,sample_size,nx*rowrange, size*size, x.shape
        ri   = 0
        for xir in range(rowrange):
            xi = row + xir
            #print xi
            for yi in range(ny):
                x1 = xi+size-h
                x2 = xi+size+h+1
                y1 = yi+size-h
                y2 = yi+size+h+1

                #print ri, x1, x2, y1, y2
                x[ri] = image[x1:x2, y1:y2].flatten()
                ri = ri +1

        whole_data=x
        if data_mean != None: 
            whole_data = whole_data - np.tile(data_mean,(np.shape(whole_data)[0],1))
        if data_std != None:    
            whole_data = whole_data / np.tile(data_std,(np.shape(whole_data)[0],1))
        return whole_data

    @staticmethod
    def generate_patch_data_rows(image, data_mean=None, data_std=None, rowOffset=0, rowRange=1, patchSize=28, imSize=(1024,1024)):
        border = np.int(np.ceil(patchSize/2.0))
        whole_set_patches = np.zeros(( imSize[1]*rowRange, patchSize*patchSize), dtype=np.float32)
        counter = 0;
        for row in xrange(rowRange):
            for col in xrange(0,imSize[1]):
                if row+rowOffset-border+1<0 or row+rowOffset+border>imSize[0] or col-border+1<0 or col+border>imSize[1]:
                    whole_set_patches[counter,:] = 0.0
                    counter+=1
                    continue
                imgPatch = image[row+rowOffset-border+1:row+rowOffset+border, col-border+1:col+border]
                #imgPatch = scale_to_unit_interval(imgPatch)
                whole_set_patches[counter,:] = imgPatch.flatten()
                counter += 1

        whole_data = np.float32(whole_set_patches)

        middlePixel = whole_data[:,np.int(np.ceil(np.shape(whole_data)[1]/2.0))]

        if data_mean != None:
            whole_data = whole_data - np.tile(data_mean,(np.shape(whole_data)[0],1))
        if data_std != None:
            whole_data = whole_data / np.tile(data_std,(np.shape(whole_data)[0],1))

        return np.float32(whole_data)#, middlePixel


    #-------------------------------------------------------------------
    # Reads the input image and normalize it.  Also creates
    # a version of the input image with corner pixels
    # mirrored based on the sample size
    # arguments: - path : directory path where image is located
    #            - id   : the unique identifier of the image
    #            - pad  : the amount to pad the image by.
    # return   : returns original image and a padded version
    #-------------------------------------------------------------------
    @staticmethod
    def get_image(img_id, img_dir):
        filename = '%s/%s.tif'%(img_dir, img_id)

        #print 'filename:', filename
        if not os.path.exists( filename ):
            print 'Error: file %s not found'%(filename)
            return False, None

        return True, Utility.read_n_scale_image( filename )


    @staticmethod
    def read_image(path, patch_size, padded=False):
        image  = tiff.imread( path )
        image  = Utility.normalizeImage( image )
        if padded:
            mode   = 'symmetric'
            pad    = patch_size
            image  = np.pad(image, ((pad, pad), (pad, pad)), mode)
        return image

    @staticmethod
    def get_image_paddedn(path, patch_size):
        image  = tiff.imread( path )
        image  = Utility.normalizeImage( image )
        mode   = 'symmetric'
        pad    = patch_size
        return np.pad(image, ((pad, pad), (pad, pad)), mode)

    @staticmethod
    def get_image_padded(path, sample_size):
        '''
        if not os.path.exists( path ):
                print 'Error: file %s not found'%(path)
                return False, None
        '''
        img     = Utility.read_n_scale_image( path )
        mode    = 'symmetric'
        pad     = sample_size
        img_pad = np.pad(img, ((pad, pad), (pad, pad)), mode)
        #return True, img_pad
        return img_pad


    #-------------------------------------------------------------------
    # Reads an image from the specified filename,
    # scales its pixel values to 0.0 .. 1.0 range,
    # and returns the scaled image to the caller
    #
    # arguments: - filename : the name of the image file to load
    #	     - bits     : the number of bits in the image
    # returns  : image with pixels scaled to 0.0 .. 1.0
    #-------------------------------------------------------------------
    @staticmethod
    def read_n_scale_image(filename, bits=8):
        img = tiff.imread(filename)
        scale = 1.0/((2**bits) - 1.0);
        return np.asarray(img)*scale;

    @staticmethod
    def create_initial_model( project, baseProject, modeltype, modelsdir ):
        if baseProject is None or baseProject == '':
            return

        modeltype = modeltype.lower()
        srcpath = '%s/best_%s_model.%s.pkl'%(modelsdir, modeltype, baseProject)
        if not os.path.exists( srcpath ):
            return

        dstpath = '%s/best_%s_model.%s.pkl'%(modelsdir, modeltype, project)
        if os.path.exists( dstpath ):
            return

        print 'copying to dst'
        shutil.copyfile( srcpath, dstpath )



    #-------------------------------------------------------------------
    # Check if there's incoming data in the specified path
    # arguments: path - the path to check
    # return   : true if has text files, false otherwise
    #-------------------------------------------------------------------
    @staticmethod
    def get_training_data(
        project,
        img_id,
        img_pad,
        sample_size,
        annotationdir):

        path = '%s/%s.%s.json'%(annotationdir, img_id, project)
        data = None

        # report error if no json file
        if not os.path.isfile( path ):
            msg = 'unable to open %s'%(path)
            Utility.report_status( msg, 'fail')
            return False, None, None

        # load the annotation data from the file
        annotations = None
        with open(path) as json_file:
            annotations = json.load( json_file )

        # report error of data was not loaded
        if annotations == None:
            msg = 'reading  %s'%(path)
            Utility.report_status( msg, 'fail')
            return False, None, None

        # extract the training data
        n_cols = sample_size * sample_size
        n_rows = 0
        #for name, coordinates in annotations.iteritems():
        for coordinates in annotations:
            n_rows += len(coordinates)/2

        # extract the sample images and labels
        x = np.zeros( (n_rows, n_cols), dtype=np.float32)
        y = np.zeros( n_rows, dtype=int )
        s = sample_size
        h = s/2
        h = (h+1) if s%2 == 0 else h

        i_row   = 0

        label_offsets = []
        label_sizes = []
        ncoordinates = 0
        for label_index, coordinates in enumerate( annotations ):

            # compute the label offset based on the number of
            # annotations of the previous label or 0 if first label
            #label_offsets.append( label_index * ncoordinates )

            ncoordinates = len(coordinates)

            #TODO: need map labels to increasing sequence 0,1,2,...
            #coordinates = annotations[ label_index ]

            #for coord in label.get('coordinates'):
            i = 0
            label_size = 0
            label_offset = i_row
            while i<ncoordinates:
                col      = coordinates[i]
                row      = coordinates[i+1]
                r1       = row+s-h
                r2       = row+s+h+1
                c1       = col+s-h
                c2       = col+s+h+1
                x[i_row] = img_pad[r1:r2, c1:c2].flatten()
                y[i_row] = label_index
                i_row   += 1
                i       += 2
                label_size += 1

            label_offsets.append( label_offset )
            label_sizes.append( label_size )
        return True, x, y, label_offsets, label_sizes


    @staticmethod
    def report_status(msg, status):
        status_msg = '%s'%(msg)
        max_chars = 80
        while (len(status_msg) + len(status)) < max_chars:
            status_msg = '%s.'%(status_msg)
        status_msg = '%s%s'%(status_msg, status)
        print status_msg

    @staticmethod
    def compare_lists(a, b):
        if len(a) != len(b):
            return False
        for i in xrange(len(a)):
            if a[i] != b[i]:
                return False
        return True

    @staticmethod
    def compress(data):
        output = StringIO.StringIO()
        output.write(data)
        content = output.getvalue()
        encoded = base64.b64encode(content)
        return zlib.compress(encoded)

    @staticmethod
    def decompress(data):
        decompressed = zlib.decompress(data)
        return base64.b64decode(decompressed)

    @staticmethod
    def getProjects():
        path = 'resources/settings.json'
        content = '{}'
        try:
            with open(path, 'r') as content_file:
                content = content_file.read()
        except:
            pass
        return Utility.compress(content)

    # web utils
    @staticmethod
    def print_msg(msg, colored=False, status=''):
        msg_len = len(msg)
        sts_len = len(status)
        dsh_len = 80 - (msg_len + sts_len)
        dsh = '.'*dsh_len
        if colored:
            dsh += '.'*9
        print '%s%s%s'%(msg,dsh,status)
