import cPickle
import gzip

import numpy as np
import mahotas
import scipy.ndimage
import scipy.misc
import skimage.transform
import glob
import random
import time
import sys
import theano
import theano.tensor as T
import shutil
from theano.tensor.shared_randomstreams import RandomStreams
#import matplotlib.pyplot as plt
import multiprocessing

import progressbar
#from utils import tile_raster_images, scale_to_unit_interval

import PIL.Image

from paths import Paths


def normalizeImage(img, saturation_level=0.05): #was 0.005
        sortedValues = np.sort( img.ravel())
        minVal = np.float32(sortedValues[len(sortedValues) * (saturation_level / 2)])
        maxVal = np.float32(sortedValues[len(sortedValues) * (1 - saturation_level / 2)])
        normImg = np.float32(img - minVal) * (255 / (maxVal-minVal))
        normImg[normImg<0] = 0
        normImg[normImg>255] = 255
        return (np.float32(normImg) / 255.0)

def write_image_as_uint16(img, fileName):
    img = np.uint16(scale_to_unit_interval(img) * (2 ** 16 -1))
    mahotas.imsave(fileName,np.uint16(img))

def shared_dataset(data_xy, borrow=True, doCastLabels=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    
    if not doCastLabels:
        shared_y = theano.shared(np.asarray(data_y,
                        dtype=theano.config.floatX),
                        borrow=borrow)
    else:
        shared_y = theano.shared(np.asarray(data_y,
                        dtype=np.int32),
                        borrow=borrow)

    return shared_x, shared_y


def watershed_adjusted_membranes(img_file_name, img_membrane_file_name):
    print 'reading image ' + img_file_name
    img = mahotas.imread(img_file_name)
    label_img = mahotas.imread(img_membrane_file_name)        
    
    blur_img = scipy.ndimage.gaussian_filter(img, 1)

    #put boundaries as either extracellular/unlabeled space or gradient between two labeled regions into one image
    boundaries = label_img==0
    boundaries[0:-1,:] = np.logical_or(boundaries[0:-1,:], np.diff(label_img, axis=0)!=0)
    boundaries[:,0:-1] = np.logical_or(boundaries[:,0:-1], np.diff(label_img, axis=1)!=0)
    
    #erode to be sure we include at least one membrane
    shrink_radius=4
    y,x = np.ogrid[-shrink_radius:shrink_radius+1, -shrink_radius:shrink_radius+1]
    shrink_disc = x*x + y*y <= (shrink_radius ** 2)
    inside = mahotas.dilate(boundaries ==0, shrink_disc)

    #use watersheds to find the actual membranes (sort of)
    seeds = label_img.copy()
    seeds[np.nonzero(inside==0)] = 0
    seeds,_ = mahotas.label(seeds == 0)
    
    wsImage = 255-np.uint8(scale_to_unit_interval(blur_img)*255)
    grow = mahotas.cwatershed(wsImage, seeds)
   
    membrane = np.zeros(img.shape, dtype=np.uint8)
    membrane[0:-1,:] = np.diff(grow, axis=0) != 0
    membrane[:,0:-1] = np.logical_or(membrane[:,0:-1], np.diff(grow,axis=1) != 0)

    return np.uint8(membrane*255)


def preprocess_images(img, label_img, adjusted_label_img):
    #first normalize gray image
    img = normalizeImage(img)
    blur_img = scipy.ndimage.gaussian_filter(img, 1)
    
    #put boundaries as either extracellular/unlabeled space or gradient between two labeled regions into one image
    boundaries = label_img==0
    boundaries[0:-1,:] = np.logical_or(boundaries[0:-1,:], np.diff(label_img, axis=0)!=0)
    boundaries[:,0:-1] = np.logical_or(boundaries[:,0:-1], np.diff(label_img, axis=1)!=0)

    #erode to be sure we include at least one membrane
    shrink_radius=5
    y,x = np.ogrid[-shrink_radius:shrink_radius+1, -shrink_radius:shrink_radius+1]
    shrink_disc = x*x + y*y <= (shrink_radius ** 2)
    inside = mahotas.erode(boundaries ==0, shrink_disc)

    non_membrane = mahotas.erode(inside, shrink_disc)
    return(adjusted_label_img, non_membrane)
    

def gen_data_supervised(makeItShort=False, purpose='train', nsamples=1000, patchSize=29, balanceRate=0.5, data_mean=0.0, data_std=1.0, grayImages=None, labelImages=None, maskImages=None, outfile = 'isbiData.pkl.gz', saveData=False):
    start_time = time.time()

    #rnd = np.random.RandomState()
   
    #pathPrefix = '/media/vkaynig/NewVolume/IAE_ISBI2012/'
    #pathPrefix = '/media/vkaynig/NewVolume/Cmor_paper_data/'
    #pathPrefix = '/home/fgonda/icon/data/reference/'
    #pathPrefix = '/n/home00/fgonda/icon/data/reference/' 
    pathPrefix = '%s/'%Paths.Reference
    #pathPrefix = '/n/home00/fgonda/icon/data/temp/'
    img_search_string_grayImages = pathPrefix + 'images/' + purpose + '/*.tif'
    #img_search_string_membraneImages = pathPrefix + 'labels/adjustedMembranes/' + purpose + '/*.tif'
    img_search_string_membraneImages = pathPrefix + 'labels/membranes/' + purpose + '/*.tif'
    img_search_string_backgroundMaskImages = pathPrefix + 'labels/background/' + purpose + '/*.tif'


    img_files_gray = sorted( glob.glob( img_search_string_grayImages ) )
    img_files_label = sorted( glob.glob( img_search_string_membraneImages ) )
    img_files_backgroundMask = sorted( glob.glob( img_search_string_backgroundMaskImages ) )

    whole_set_patches = np.zeros((nsamples, patchSize*patchSize), dtype=np.float)
    whole_set_labels = np.zeros(nsamples, dtype=np.int32)


    print 'src:', img_search_string_grayImages
    print 'mem:', img_search_string_membraneImages
    print 'label:', img_search_string_membraneImages
    print 'msk:', img_search_string_backgroundMaskImages
    #print 'lbs:', img_files_label
    #print 'nsamples:', nsamples
    #print img_files_gray

#how many samples per image?
    nsamples_perImage = np.uint(np.ceil( 
            (nsamples) / np.float(np.shape(img_files_gray)[0])
            )) 
    print 'using ' + np.str(nsamples_perImage) + ' samples per image.'
    counter = 0
#    bar = progressbar.ProgressBar(maxval=nsamples, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    if grayImages is None:
        img = mahotas.imread(img_files_gray[0])
        grayImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
        labelImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
        maskImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))

        for img_index in xrange(np.shape(img_files_gray)[0]):
                img = mahotas.imread(img_files_gray[img_index])
                img = normalizeImage(img) 
                grayImages[:,:,img_index] = img
                label_img = mahotas.imread(img_files_label[img_index])        
                labelImages[:,:,img_index] = label_img
                mask_img = mahotas.imread(img_files_backgroundMask[img_index])
                maskImages[:,:,img_index] = mask_img
            
    for img_index in xrange(np.shape(img_files_gray)[0]):
        img = grayImages[:,:,img_index]        
        label_img = labelImages[:,:,img_index]
        mask_img = maskImages[:,:,img_index]

        #additionally mask pixels that are too bright to be membrane anyways
#        mask_dark = img < 0.7
#        mask_img = np.logical_and(mask_img, mask_dark)

        #get rid of invalid image borders
        border_patch = np.ceil(patchSize/2.0)
        #border = np.ceil(patchSize/2.0)
        border = np.ceil(np.sqrt(2*(border_patch**2)))
        label_img[:border,:] = 0 #top
        label_img[-border:,:] = 0 #bottom
        label_img[:,:border] = 0 #left
        label_img[:,-border:] = 0 #right

        mask_img[:border,:] = 0
        mask_img[-border:,:] = 0
        mask_img[:,:border] = 0
        mask_img[:,-border:] = 0

        membrane_indices = np.nonzero(label_img)
        non_membrane_indices = np.nonzero(mask_img)

        positiveSample = True
        for i in xrange(nsamples_perImage):
            if counter >= nsamples:
                break
#            positiveSample = rnd.random() < balanceRate

            if positiveSample:
                randmem = random.choice(xrange(len(membrane_indices[0])))
                (row,col) = (membrane_indices[0][randmem], 
                             membrane_indices[1][randmem])
                label = 1.0
                positiveSample = False
            else:
                randmem = random.choice(xrange(len(non_membrane_indices[0])))
                (row,col) = (non_membrane_indices[0][randmem], 
                             non_membrane_indices[1][randmem])
                label = 0.0
                positiveSample = True
                    
            imgPatch = img[row-border+1:row+border, col-border+1:col+border]
#            print "patch: ", np.max(imgPatch)
            #imgPatch = scipy.misc.imrotate(imgPatch,random.choice(xrange(360)))
            imgPatch = skimage.transform.rotate(imgPatch, random.choice(xrange(360)))
#            print "after rotation: ", np.max(imgPatch)
            imgPatch = imgPatch[border-border_patch+1:border+border_patch,border-border_patch+1:border+border_patch]
#            print "small patch: ", np.max(imgPatch)
#            middle = np.floor(patchSize/2.0)
#            blindSpotSize = 6
#            imgPatch[middle-blindSpotSize:middle+blindSpotSize,middle-blindSpotSize:middle+blindSpotSize] = 0.0

            if random.random() < 0.5:
                    imgPatch = np.fliplr(imgPatch)
#                    print "after flip: ", np.max(imgPatch)

            #Force network to learn shapes instead of gray values by inverting randomly
            if True: #random.random() < 0.5:
                    whole_set_patches[counter,:] = imgPatch.flatten()
            else:
                    whole_set_patches[counter,:] = 1-imgPatch.flatten()
            
            whole_set_labels[counter] = label
            counter += 1
#            bar.update(counter)

#    bar.finish()     
    #normalize data
    whole_data = np.float32(whole_set_patches)
#    print "whole_data: ", np.max(whole_data)    
    if data_mean == None:
        whole_data_mean = np.mean(whole_data,axis=0)
    else:
        whole_data_mean = data_mean

    whole_data = whole_data - np.tile(whole_data_mean,(np.shape(whole_data)[0],1))
#    print "whole_data normalized mean: ", np.max(whole_data)        

#    whole_data_mean = np.mean(whole_data, axis=1)
#    whole_data = whole_data - np.transpose(np.tile(whole_data_mean, (np.shape(whole_data)[1],1)))

    if data_std == None:
        whole_data_std = np.std(whole_data,axis=0)
    else:
        whole_data_std = data_std

    whole_data_std = np.clip(whole_data_std, 0.00001, np.max(whole_data_std))
    whole_data = whole_data / np.tile(whole_data_std,(np.shape(whole_data)[0],1))

    #data = np.uint8(whole_data*255).copy()
    #labels = np.uint8(whole_set_labels).copy()

    data = whole_data.copy()
    labels = whole_set_labels.copy()

    #remove the sorting in image order
    shuffleIndex = np.random.permutation(np.shape(labels)[0])
    for i in xrange(np.shape(labels)[0]):  
        whole_data[i,:] = data[shuffleIndex[i],:]
        whole_set_labels[i] = labels[shuffleIndex[i]]
    
    data_set = (whole_data, whole_set_labels)    

    print np.max(data_set[0])
    end_time = time.time()
    total_time = (end_time - start_time)
    print >> sys.stderr, ('Running time: ' +
                          '%.2fm' % (total_time / 60.))
#Save the results
    if saveData:
        print 'Saving data.'
        f = gzip.open(outfile,'wb', compresslevel=1)
        #f = open(outfile,'wb')
        cPickle.dump(data_set,f)
        f.close()
        f = gzip.open('normCoefs_'+outfile,'wb', compresslevel=1)
        #f = open('normCoefs_'+outfile,'wb')
        cPickle.dump((whole_data_mean, whole_data_std),f)
        f.close()
        
        print "Saved"


    rval = data_set
    print 'finished sampling data'

    if makeItShort:
            return rval
    else:
            return (rval, whole_data_mean, whole_data_std, grayImages, labelImages, maskImages)


def get_patch_data_for_image(img, label_img, mask_img, 
                             nsamples_perImage, patchSize, rnd, bar):
    
    this_set_patches = np.zeros((nsamples_perImage, patchSize*patchSize), dtype=np.float)
    this_set_labels = np.zeros(nsamples_perImage, dtype=np.int32)
    counter = 0

    #additionally mask pixels that are too bright to be membrane anyways
    #        mask_dark = img < 0.7
    #        mask_img = np.logical_and(mask_img, mask_dark)

    #get rid of invalid image borders
    border_patch = np.ceil(patchSize/2.0)
    #border = np.ceil(patchSize/2.0)
    border = np.ceil(np.sqrt(2*(border_patch**2)))
    label_img[:border,:] = 0 #top
    label_img[-border:,:] = 0 #bottom
    label_img[:,:border] = 0 #left
    label_img[:,-border:] = 0 #right
    
    mask_img[:border,:] = 0
    mask_img[-border:,:] = 0
    mask_img[:,:border] = 0
    mask_img[:,-border:] = 0

    membrane_indices = np.nonzero(label_img)
    non_membrane_indices = np.nonzero(mask_img)
    
    positiveSample = True
    for i in xrange(nsamples_perImage):
        if positiveSample:
            randmem = rnd.choice(xrange(len(membrane_indices[0])))
            (row,col) = (membrane_indices[0][randmem], 
                         membrane_indices[1][randmem])
            label = 1.0
            positiveSample = False
        else:
            randmem = rnd.choice(xrange(len(non_membrane_indices[0])))
            (row,col) = (non_membrane_indices[0][randmem], 
                         non_membrane_indices[1][randmem])
            label = 0.0
            positiveSample = True
            
        imgPatch = img[row-border+1:row+border, col-border+1:col+border]
        #imgPatch = scipy.misc.imrotate(imgPatch,random.choice(xrange(360)))
        imgPatch = skimage.transform.rotate(imgPatch, random.choice(xrange(360)))
        imgPatch = imgPatch[border-border_patch+1:border+border_patch,border-border_patch+1:border+border_patch]
#            middle = np.floor(patchSize/2.0)
#            blindSpotSize = 6
#            imgPatch[middle-blindSpotSize:middle+blindSpotSize,middle-blindSpotSize:middle+blindSpotSize] = 0.0

        if random.random() < 0.5:
            imgPatch = np.fliplr(imgPatch)

            #Force network to learn shapes instead of gray values by inverting randomly
        if True: #random.random() < 0.5:
                this_set_patches[counter,:] = imgPatch.flatten()
        else:
                this_set_patches[counter,:] = 1-imgPatch.flatten()
                    
        this_set_labels[counter] = label
        counter += 1
    return (this_set_patches, this_set_labels)

def stupid_map_wrapper(parameters):
        f = parameters[0]
        args = parameters[1:]
        return f(*args)

def gen_data_supervised_parallel(purpose='train', nsamples=1000, patchSize=29, outfile = 'isbiData.pkl.gz', saveData=False, balanceRate=0.5, data_mean=0.0, data_std=1.0, grayImages=None, labelImages=None, maskImages=None, nrProcesses=4):

    start_time = time.time()

    rnd = random.Random()
    #pathPrefix = '/media/vkaynig/NewVolume/IAE_ISBI2012/'
    pathPrefix = '/media/vkaynig/NewVolume/Cmor_paper_data/'
    img_search_string_grayImages = pathPrefix + 'images/' + purpose + '/*.tif'
    img_search_string_membraneImages = pathPrefix + 'labels/adjustedMembranes/' + purpose + '/*.tif'
    #img_search_string_membraneImages = pathPrefix + 'labels/membranes/' + purpose + '/*.tif'
    img_search_string_backgroundMaskImages = pathPrefix + 'labels/background/' + purpose + '/*.tif'

    img_files_gray = sorted( glob.glob( img_search_string_grayImages ) )
    img_files_label = sorted( glob.glob( img_search_string_membraneImages ) )
    img_files_backgroundMask = sorted( glob.glob( img_search_string_backgroundMaskImages ) )

    whole_set_patches = np.zeros((nsamples, patchSize*patchSize), dtype=np.float)
    whole_set_labels = np.zeros(nsamples, dtype=np.int32)

    #how many samples per image?
    nsamples_perImage = np.uint(np.ceil( 
            (nsamples) / np.float(np.shape(img_files_gray)[0])
            )) 
    print 'using ' + np.str(nsamples_perImage) + ' samples per image.'

    bar = progressbar.ProgressBar(maxval=np.shape(img_files_gray)[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    if grayImages is None:
        print "Reading images."
        img = mahotas.imread(img_files_gray[0])
        grayImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
        labelImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
        maskImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))

        for img_index in xrange(np.shape(img_files_gray)[0]):
                img = mahotas.imread(img_files_gray[img_index])
                img = normalizeImage(img) 
                grayImages[:,:,img_index] = img
                label_img = mahotas.imread(img_files_label[img_index])        
                labelImages[:,:,img_index] = label_img
                mask_img = mahotas.imread(img_files_backgroundMask[img_index])
                maskImages[:,:,img_index] = mask_img
            
        print "Done."
#    for img_index in xrange(np.shape(img_files_gray)[0]):
#        patches, labels = get_patch_data_for_image(img_index, grayImages, labelImages, maskImages, 
#                                                   nsamples_perImage, patchSize, rnd, bar)
#        whole_set_patches[img_index*nsamples_perImage:(img_index+1)*nsamples_perImage,:] = patches
#        whole_set_labels[img_index*nsamples_perImage:(img_index+1)*nsamples_perImage] = labels
#        result = (0,0,0,0,0,0)
#
    print "Starting pool."
    pool = multiprocessing.Pool(processes=nrProcesses)
    result = pool.map(stupid_map_wrapper, [(get_patch_data_for_image, grayImages[:,:,img_index], labelImages[:,:,img_index], 
                                            maskImages[:,:,img_index], nsamples_perImage, 
                                            patchSize, rnd, bar) for img_index in range(np.shape(img_files_gray)[0])])

    pool.close()
    pool.join() 
    print "Pool closed."

#    result = result.get()

    whole_set_patches = np.vstack([p[0] for p in result])
    whole_set_labels = np.hstack([p[1] for p in result])
    
    bar.finish()    
    #normalize data
    whole_data = np.float32(whole_set_patches)
    if data_mean == None:
        whole_data_mean = np.mean(whole_data,axis=0)
    else:
        whole_data_mean = data_mean

    whole_data = whole_data - np.tile(whole_data_mean,(np.shape(whole_data)[0],1))

    if data_std == None:
        whole_data_std = np.std(whole_data,axis=0)
    else:
        whole_data_std = data_std

    whole_data_std = np.clip(whole_data_std, 0.00001, np.max(whole_data_std))
    whole_data = whole_data / np.tile(whole_data_std,(np.shape(whole_data)[0],1))

    #data = np.uint8(whole_data*255).copy()
    #labels = np.uint8(whole_set_labels).copy()

    data = whole_data.copy()
    labels = whole_set_labels.copy()

    #remove the sorting in image order
    shuffleIndex = np.random.permutation(np.shape(labels)[0])
    for i in xrange(np.shape(labels)[0]):  
        whole_data[i,:] = data[shuffleIndex[i],:]
        whole_set_labels[i] = labels[shuffleIndex[i]]
    
    data_set = (whole_data, whole_set_labels)    

    print np.max(data_set[0])
    end_time = time.time()
    total_time = (end_time - start_time)
    print >> sys.stderr, ('Running time: ' +
                          '%.2fm' % (total_time / 60.))
    rval = data_set
    print 'finished sampling data'

    return (rval, whole_data_mean, whole_data_std, grayImages, labelImages, maskImages)



def generate_experiment_data_unsupervised(ntrain=6000, nvalid=1000, ntest=1000, patchSize=21,outfile = 'isbiData.pkl.gz', saveData=True, data_mean=None, data_std=None):
    start_time = time.clock()
    img_search_string_grayImages = '/data/Verena/I*_image.tif'

    img_files_gray = sorted( glob.glob( img_search_string_grayImages ) )

    whole_set_patches = np.zeros((ntrain+nvalid+ntest, patchSize*patchSize), dtype=np.float)
    whole_set_patches_noisy = np.zeros((ntrain+nvalid+ntest, patchSize*patchSize), dtype=np.float)

#how many samples per image?
    nsamples = np.uint(np.ceil( 
            (ntrain+nvalid+ntest) / np.float(np.shape(img_files_gray)[0])
            )) 
    print 'using ' + np.str(nsamples) + ' samples per image.'
    counter = 0

    for img_index in xrange(np.shape(img_files_gray)[0]):
        img = mahotas.imread(img_files_gray[img_index])
        img = normalizeImage(img)

        #get rid of invalid image borders
        border_patch = np.ceil(patchSize/2.0)
        border = np.ceil(np.sqrt(2*(border_patch**2)))

        mask_img = np.ones(np.shape(img))
        mask_img[:border,:] = 0
        mask_img[-border:,:] = 0
        mask_img[:,:border] = 0
        mask_img[:,-border:] = 0

        indices = np.nonzero(mask_img)

        for i in xrange(nsamples):
            randmem = random.choice(xrange(len(indices[0])))
            (row,col) = (indices[0][randmem], 
                         indices[1][randmem])
                
            imgPatch = img[row-border+1:row+border, col-border+1:col+border]
            imgPatch = scale_to_unit_interval(scipy.misc.imrotate(imgPatch,random.choice(range(360))))
            imgPatch = imgPatch[border-border_patch+1:border+border_patch,border-border_patch+1:border+border_patch]

            whole_set_patches[counter,:] = imgPatch.flatten() #flatten creates a copy
            
            #now induce noise
            imgPatch_data = imgPatch[0::5] #only keep every 6th row.. projection might be better model here
            ps = np.shape(imgPatch_data)
            imgPatch_noisy = np.reshape(np.tile(imgPatch_data,5),(ps[0]*5, ps[1]))

            ps = np.ceil(np.shape(imgPatch)[0]/2)
            center = np.ceil(np.shape(imgPatch_noisy)[0]/2.0)

            imgPatch_noisy = imgPatch_noisy[center-ps:center+ps+1,:]
            
            whole_set_patches_noisy[counter,:] = imgPatch_noisy.flatten() 
            counter += 1

    #normalize data
    whole_data = whole_set_patches

    whole_data_mean = np.mean(whole_data, axis=1)
    whole_data = whole_data - np.transpose(np.tile(whole_data_mean, (np.shape(whole_data)[1],1)))

#now split into train val and test set
    train_set = (np.uint8(whole_data[0:ntrain,:]*255), np.uint8(whole_set_patches_noisy[0:ntrain,:]))
    valid_set = (np.uint8(whole_data[ntrain:ntrain+nvalid,:]*255), np.uint8(whole_set_patches_noisy[ntrain:ntrain+nvalid,:])) 
    test_set = (np.uint8(whole_data[ntrain+nvalid:ntrain+nvalid+ntest,:]*255), np.uint8(whole_set_patches_noisy[ntrain+nvalid:ntrain+nvalid+ntest,:]))

    end_time = time.clock()
    total_time = (end_time - start_time)
    print >> sys.stderr, ('Running time: ' +
                          '%.2fm' % (total_time / 60.))


#Save the results
    if saveData:
        print 'Saving data.'
        f = gzip.open(outfile,'wb', compresslevel=1)
        cPickle.dump((train_set, valid_set, test_set),f)
        f.close()
        
        print "Saved"


    test_set_x, test_set_y = shared_dataset(test_set, doCastLabels=False)
    valid_set_x, valid_set_y = shared_dataset(valid_set, doCastLabels=False)
    train_set_x, train_set_y = shared_dataset(train_set, doCastLabels=False)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
    print 'finished sampling data'

    return (rval, data_mean, data_std)

def visualize_examples_supervised(dataset):
    patches = dataset[0]
    labels = dataset[1]

    ps = np.sqrt(np.shape(patches)[1])

    posIndices = np.nonzero(labels)
    negIndices = np.nonzero(1-labels)
    
    imagePos = PIL.Image.fromarray(tile_raster_images(X=patches[posIndices], img_shape=(ps,ps), tile_shape=(10,10), tile_spacing=(1,1)))
    imageNeg = PIL.Image.fromarray(tile_raster_images(X=patches[negIndices], img_shape=(ps,ps), tile_shape=(10,10), tile_spacing=(1,1)))

    '''
    plt.subplot(2,1,1)
    plt.imshow(imagePos)
    plt.subplot(2,1,2)
    plt.imshow(imageNeg)
    plt.show()
    '''


def write_patch_examples_supervised(dataset):
    patches = dataset[0]
    labels = dataset[1]

    ps = np.sqrt(np.shape(patches)[1])

    posIndices = np.nonzero(labels)
    negIndices = np.nonzero(1-labels)
    
    imagePos = PIL.Image.fromarray(tile_raster_images(X=patches[posIndices], img_shape=(ps,ps), tile_shape=(10,10), tile_spacing=(1,1)))

    imageNeg = PIL.Image.fromarray(tile_raster_images(X=patches[negIndices], img_shape=(ps,ps), tile_shape=(10,10), tile_spacing=(1,1)))

    imagePos.save("some_positiveExamples.png")
    imageNeg.save("some_negativeExamples.png")

def write_patch_examples_unsupervised(datasets):
    patches = datasets[0][0].get_value()
    ps = np.sqrt(np.shape(patches)[1])
    image = PIL.Image.fromarray(tile_raster_images(X=patches, img_shape=(ps,ps), tile_shape=(10,10), tile_spacing=(1,1)))

    image.save("some_originalExamples.png")
    
    patches = datasets[0][1].get_value()
    ps = np.sqrt(np.shape(patches)[1])
    image = PIL.Image.fromarray(tile_raster_images(X=patches, img_shape=(ps,ps), tile_shape=(10,10), tile_spacing=(1,1)))

    image.save("some_noisyExamples.png")

if __name__ == '__main__':      
        print "not implemented"

