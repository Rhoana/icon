import cPickle
import gzip

import numpy as np
import mahotas
import glob
import random
import time
import sys
import theano
import theano.tensor as T
from theano.tensor.signal.conv import conv2d
from theano.tensor.shared_randomstreams import RandomStreams
from generateTrainValTestData import normalizeImage
#from utils import tile_raster_images, scale_to_unit_interval
#from vsk_utils import shared_single_dataset

import PIL.Image


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


def classifyImage(imageName, network_layerList):

    def applyNetwork(data):
        for da in network_layerList[:-1]:
            data = da.get_hidden_values(data)

        p_y_given_x = network_layerList[-1].get_p_y_given_x(data)
        return p_y_given_x

    start_time = time.clock()

    row_range = 1
    img = mahotas.imread(imageName)
    img = normalizeImage(img)

    imSize = np.shape(img)

    membraneProbabilities = []
    patchSize = np.int(np.sqrt(network_layerList[0].n_visible))

    data_type = T.matrix('data')

    classifiedData = applyNetwork(data_type)
    classify = theano.function(inputs=[data_type], outputs=classifiedData)

    for row in xrange(0,1024,row_range):
        if row%100 == 0:
            print row
        data = generate_patch_data_rows(img, rowOffset=row, rowRange=row_range, patchSize=patchSize, imSize=imSize)
        result = classify(data)
        membraneProbabilities.append(result[:,1])

    end_time = time.clock()
    total_time = (end_time - start_time)
    print >> sys.stderr, ('Running time: ' +
                          '%.2fm' % (total_time / 60.))

    return np.array(membraneProbabilities)


def save_classification(fileName, prediction):
    PIL.Image.fromarray(np.uint8(prediction*255)).save(fileName)

def classifyImage_MLP(imageName, classifier, data_mean=None, data_std=None, doThresh=False):

    start_time = time.clock()

    row_range = 1
    img = mahotas.imread(imageName)
    img = normalizeImage(img)

    if doThresh:
        mask = img >= 0.7

    imSize = np.shape(img)

    membraneProbabilities = []
    patchSize = np.int(np.sqrt(classifier.n_visible))

    data_type = T.matrix('data')

    classify = classifier.buildClassification_function(data_type)

    for row in xrange(0,1024,row_range):
        if row%100 == 0:
            print row
        data, middlePixel = generate_patch_data_rows(img, data_mean, data_std, rowOffset=row, rowRange=row_range, patchSize=patchSize, imSize=imSize)
        result = classify(data)
        result = result[:,1]
        if doThresh:
            result[middlePixel>0.7]=0.0
        membraneProbabilities.append(result)

    end_time = time.clock()
    total_time = (end_time - start_time)
    print >> sys.stderr, ('Running time: ' +
                          '%.2fm' % (total_time / 60.))

    if doThresh:
        membraneProbabilities = np.array(membraneProbabilities)
        membraneProbabilities[mask>0] = np.min(membraneProbabilities)
    return np.array(membraneProbabilities)


def classifyImage_MLP_iae(imageName, classifier, iae):

    start_time = time.clock()

    row_range = 1
    img = mahotas.imread(imageName)
    #img = np.float32(img)
    #img = img - img.min()
    #img = img / img.max()
    img = normalizeImage(img)

    imSize = np.shape(img)

    membraneProbabilities = []
    patchSize = np.int(np.sqrt(classifier.n_visible))

    data_type = T.matrix('data')

    get_first_layer_code = classifier.get_layer_output_function(data=data_type, layerNumber=0)
    applyIAE = iae.buildClassification_function(data=data_type)
    classify_from_code = classifier.classify_from_code(code=data_type, layerNumber=1)

    for row in xrange(0,1024,row_range):
        if row%100 == 0:
            print row
        data = generate_patch_data_rows(img, rowOffset=row, rowRange=row_range, patchSize=patchSize, imSize=imSize)
        intermediate_code = get_first_layer_code(data)
        corrected_code = applyIAE(intermediate_code)
        result = classify_from_code(corrected_code)

        membraneProbabilities.append(result[:,1])

    end_time = time.clock()
    total_time = (end_time - start_time)
    print >> sys.stderr, ('Running time: ' +
                          '%.2fm' % (total_time / 60.))

    return np.array(membraneProbabilities)

#yz means row wise
def classifyImage_aniso_MLP_iae(imageName, classifier, iae_layer=None, data_mean=None, data_std=None, doThresh=False, iae=None, yz=True):

    start_time = time.clock()

    img = mahotas.imread(imageName)
    #img = np.float32(np.array(img))
    #img = img - img.min()
    #img = img / img.max()
    img = normalizeImage(img)

    if not yz:
        img = np.rot90(img,3)


    imSize = np.shape(img)
    print imSize

    membraneProbabilities = []
    patchSize = np.int(np.sqrt(classifier.n_visible))

    data_type = T.matrix('data')

    get_first_layer_code = classifier.get_layer_output_function(data=data_type, layerNumber=0)
    if not iae == None:
        applyIAE = iae.buildClassification_function(data=data_type)
    classify_from_code = classifier.classify_from_code(code=data_type, layerNumber=1)

    #for row in xrange(0,np.shape(img)[0],5):
    for row in xrange(0,170,5):
        if row%100 == 0:
            print row
        data, middlePixel = generate_patch_data_rows(image=img, data_mean=data_mean, data_std=data_std, rowOffset=row, rowRange=1, patchSize=patchSize, imSize=imSize)

        intermediate_code = get_first_layer_code(data)
        if not iae==None:
            corrected_code = applyIAE(intermediate_code)
            result = classify_from_code(corrected_code)
        else:
            result = classify_from_code(intermediate_code)

        result = result[:,1]
        if doThresh:
            result[middlePixel>0.7]=0.0
        membraneProbabilities.append(result)

    print np.shape(membraneProbabilities)
    end_time = time.clock()
    total_time = (end_time - start_time)
    print >> sys.stderr, ('Running time: ' +
                          '%.2fm' % (total_time / 60.))
    membraneProbabilities = np.array(membraneProbabilities)

    if not yz:
        membraneProbabilities = np.rot90(membraneProbabilities,1)

    return membraneProbabilities


#yz means row wise
def classifyImage_aniso(imageName, network_layerList, yz=True):

    def applyNetwork(data):
        for da in network_layerList[:-1]:
            data = da.get_hidden_values(data)

        p_y_given_x = network_layerList[-1].get_p_y_given_x(data)
        return p_y_given_x

    start_time = time.clock()

    img = mahotas.imread(imageName)
    #img = np.float32(np.array(img))
    #img = img - img.min()
    #img = img / img.max()
    img = normalizeImage(img)

    if not yz:
        img = np.rot90(img,1)


    imSize = np.shape(img)
    print imSize

    membraneProbabilities = []
    patchSize = np.int(np.sqrt(network_layerList[0].n_visible))

    data_type = T.matrix('data')

    classifiedData = applyNetwork(data_type)
    classify = theano.function(inputs=[data_type], outputs=classifiedData)

    for row in xrange(0,np.shape(img)[0],5):
        if row%100 == 0:
            print row
        data = generate_patch_data_rows(img, rowOffset=row, rowRange=1, patchSize=patchSize, imSize=imSize)
        result = classify(data)
        membraneProbabilities.append(result[:,1])

    print np.shape(membraneProbabilities)
    end_time = time.clock()
    total_time = (end_time - start_time)
    print >> sys.stderr, ('Running time: ' +
                          '%.2fm' % (total_time / 60.))
    membraneProbabilities = np.array(membraneProbabilities)

    if not yz:
        membraneProbabilities = np.rot90(membraneProbabilities,3)

    return membraneProbabilities

