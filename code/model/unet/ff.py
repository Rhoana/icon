import os
import sys
import skimage.transform
import skimage.exposure
import time
import glob
import numpy as np
import mahotas
import random
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
import json
from scipy.ndimage.filters import maximum_filter

base_path = os.path.dirname(__file__)
sys.path.insert(1,os.path.join(base_path, '../../common'))
sys.path.insert(2,os.path.join(base_path, '../../database'))

from utility import Utility
from settings import Paths
from project import Project
from paths import Paths
from db import DB

# the idea is to grow the labels to cover the whole membrane
# image and label should be [0,1]
def adjust_imprecise_boundaries(image, label, number_iterations=5):
    label = label.copy()
    label_orig = label.copy()

    for i in xrange(number_iterations):
        # grow labels by one pixel
        label = maximum_filter(label, 2)
        # only keep pixels that are on dark membrane
        non_valid_label = np.logical_and(label==1, image>0.7)
        label[non_valid_label] = 0

    # make sure original labels are preserved
    label = np.logical_or(label==1, label_orig==1)

    return label


def deform_images(image1, image2, image3=None):
    # assumes image is uint8
    def apply_deformation(image, coordinates):
        # ndimage expects uint8 otherwise introduces artifacts. Don't ask me why, its stupid.
        deformed = scipy.ndimage.map_coordinates(image, coordinates, mode='reflect')
        deformed = np.reshape(deformed, image.shape) 
        return deformed

    if np.max(image1) < 1.1:
        image1 = np.uint8(image1*255)
        image2 = np.uint8(image2*255)
        if not image3 is None:
            image3 = np.uint8(image3*255)

    displacement_x = np.random.normal(size=image1.shape, scale=10)
    displacement_y = np.random.normal(size=image1.shape, scale=10)
    
    # smooth over image
    coords_x, coords_y = np.meshgrid(np.arange(0,image1.shape[0]), np.arange(0,image1.shape[1]), indexing='ij')

    displacement_x = coords_x.flatten() + scipy.ndimage.gaussian_filter(displacement_x, sigma=5).flatten()
    displacement_y = coords_y.flatten() + scipy.ndimage.gaussian_filter(displacement_y, sigma=5).flatten()
    
    coordinates = np.vstack([displacement_x, displacement_y])
    
    deformed1 = apply_deformation(image1, coordinates) / 255.0
    deformed2 = apply_deformation(image2, coordinates) / 255.0
    if not image3 is None:
        deformed3 = apply_deformation(image3, coordinates) 
        return (deformed1, deformed2, deformed3)

    return (deformed1, deformed2)


def deform_images_list(images):
    # assumes image is uint8
    def apply_deformation(image, coordinates):
        # ndimage expects uint8 otherwise introduces artifacts. Don't ask me why, its stupid.
        deformed = scipy.ndimage.map_coordinates(image, coordinates, mode='reflect')
        deformed = np.reshape(deformed, image.shape) 
        return deformed

    displacement_x = np.random.normal(size=images.shape[:2], scale=10)
    displacement_y = np.random.normal(size=images.shape[:2], scale=10)

    # smooth over image
    coords_x, coords_y = np.meshgrid(np.arange(0,images.shape[0]), np.arange(0,images.shape[1]), indexing='ij')

    displacement_x = coords_x.flatten() #+ scipy.ndimage.gaussian_filter(displacement_x, sigma=5).flatten()
    displacement_y = coords_y.flatten() #+ scipy.ndimage.gaussian_filter(displacement_y, sigma=5).flatten()
    
    coordinates = np.vstack([displacement_x, displacement_y])
    
    deformed = images.copy()
    for i in xrange(images.shape[2]):
        deformed[:,:,i] = apply_deformation(np.uint8(images[:,:,i]), coordinates)
    
    return deformed


def normalizeImage(img, saturation_level=0.05, doClahe=False): #was 0.005
    if not doClahe:
        sortedValues = np.sort( img.ravel())
        minVal = np.float32(sortedValues[np.int(len(sortedValues) * (saturation_level / 2))])
        maxVal = np.float32(sortedValues[np.int(len(sortedValues) * (1 - saturation_level / 2))])
        normImg = np.float32(img - minVal) * (255 / (maxVal-minVal))
        normImg[normImg<0] = 0
        normImg[normImg>255] = 255
        output = (np.float32(normImg) / 255.0)
        return output
    else:
        output = skimage.exposure.equalize_adapthist(img)
        return output
    
    
def generate_experiment_data_supervised(purpose='train', nsamples=1000, patchSize=29, balanceRate=0.5, rng=np.random):
    start_time = time.time()

    if os.path.exists('/media/vkaynig/Data1/Cmor_paper_data/'):
        pathPrefix = '/media/vkaynig/Data1/Cmor_paper_data/'
    else:
        pathPrefix = '/n/pfister_lab/vkaynig/'

    img_search_string_membraneImages = pathPrefix + 'labels/membranes_nonDilate/' + purpose + '/*.tif'
    img_search_string_backgroundMaskImages = pathPrefix + 'labels/background_nonDilate/' + purpose + '/*.tif'
    
    img_search_string_grayImages = pathPrefix + 'images/' + purpose + '/*.tif'
    
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
    counter = 0
    
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
        
        #get rid of invalid image borders
        border_patch = np.int(np.ceil(patchSize/2.0))
        border = np.int(np.ceil(np.sqrt(2*(border_patch**2))))
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
            imgPatch = skimage.transform.rotate(imgPatch, random.choice(xrange(360)))
            imgPatch = imgPatch[border-border_patch:border+border_patch-1,border-border_patch:border+border_patch-1]

            if random.random() < 0.5:
                imgPatch = np.fliplr(imgPatch)
            imgPatch = np.rot90(imgPatch, random.randint(0,3))
                
            whole_set_patches[counter,:] = imgPatch.flatten()
            whole_set_labels[counter] = label
            counter += 1
            
    #normalize data
    whole_data = np.float32(whole_set_patches)
    
    whole_data = whole_data - 0.5
    
    data = whole_data.copy()
    labels = whole_set_labels.copy()
    
    #remove the sorting in image order
    shuffleIndex = rng.permutation(np.shape(labels)[0])
    for i in xrange(np.shape(labels)[0]):  
        whole_data[i,:] = data[shuffleIndex[i],:]
        whole_set_labels[i] = labels[shuffleIndex[i]]
        
    data_set = (whole_data, whole_set_labels)    
    
    end_time = time.time()
    total_time = (end_time - start_time)
    print 'Running time: ' + '%.2fm' % (total_time / 60.)
    rval = data_set
    return rval


def generate_image_data(img, patchSize=29, rows=1):
    img = normalizeImage(img) 

    # pad image borders
    border = np.int(np.ceil(patchSize/2.0))
    img_padded = np.pad(img, border, mode='reflect')

    whole_set_patches = np.zeros((len(rows)*img.shape[1], patchSize**2))

    counter = 0
    for row in rows:
        for col in xrange(img.shape[1]):
            imgPatch = img_padded[row+1:row+2*border, col+1:col+2*border]
            whole_set_patches[counter,:] = imgPatch.flatten()
            counter += 1

    #normalize data
    whole_set_patches = np.float32(whole_set_patches)
    whole_set_patches = whole_set_patches - 0.5

    return whole_set_patches


def stupid_map_wrapper(parameters):
        f = parameters[0]
        args = parameters[1:]
        return f(*args)



def gen_annotated_image(path, dim):
    image = np.zeros( (dim[0], dim[1]) )

    # assumes 
    image[:,:] = 0 

    annotations = []

    # load the annotations
    with open( path ) as labels_f:
        annotations = json.load( labels_f )

    n_labels = len(annotations)

    if n_labels == 0:
        return

    for i_label in range(n_labels):
        i_coord = 0
        coordinates = annotations[ i_label ]
        for i in range(0, len(coordinates), 2):
            x = min(coordinates[i], dim[1]-1)
            y = min(coordinates[i+1], dim[0]-1)
            image[x][y] = i_label + 1

    return image, annotations


def gen_training_data(project, nsamples=1000, patchSize=29, outPatchSize=1):
    def relabel(image):
        id_list = np.unique(image)
        for index, id in enumerate(id_list):
            image[image==id] = index
        return image

    print 'gen_data'
    if project == None:
        return

    n_labels = len( project.labels )

    start_time = time.time()

    files_gray = []
    files_annotations = []

    images = DB.getTrainingImages( project.id, new=False )
    path = Paths.TrainGrayscale

    # build the list of images to sample from while discarding those
    # without annnotations.
    for image in images:
        d_path = '%s/%s.tif'%(path, image.id)
        m_path = '%s/%s.%s.json'%(Paths.Labels, image.id, project.id)

        if os.path.exists( d_path ) and os.path.exists( m_path ):
            files_gray.append( d_path )
            files_annotations.append( m_path )


    whole_set_patches = np.zeros((nsamples, patchSize**2), dtype=np.float)
    whole_set_labels = np.zeros((nsamples, outPatchSize**2), dtype=np.int32)

    # return nothing if images or annotations not found
    if len( files_gray ) == 0 or len( files_annotations ) == 0:
        return None

    print files_gray
    print files_annotations

    whole_set_patches = np.zeros((nsamples, patchSize**2), dtype=np.float)
    whole_set_labels = np.zeros((nsamples, outPatchSize**2), dtype=np.int32)

    #how many samples per image?
    nsamples_perImage = np.uint(np.ceil( (nsamples) / np.float(np.shape(files_gray)[0]) ))
    print 'using ' + np.str(nsamples_perImage) + ' samples per image.'
    counter = 0


    # pad image borders
    border = np.int(np.ceil(patchSize/2.0))
    pad = patchSize
    
    n_samples_remaining = nsamples
    n_images = len(files_gray)

    n_samples_per_image = int(nsamples/n_images)
    n_samples_per_label = [ int(nsamples/n_labels) for label in project.labels]

    print 'n_samples_per_image:', n_samples_per_image
    print 'n_samples_per_label:', n_samples_per_label

    for i_image in range( n_images ):

        img = mahotas.imread(files_gray[ i_image ])
        ann, annotations = gen_annotated_image( files_annotations[ i_image ], img.shape )

        img = np.pad(img, ((pad, pad), (pad, pad)), 'symmetric')
        img = normalizeImage(img, doClahe=True)
        
        # get the label indices
        #indices = np.nonzero( ann )


        ann = np.pad(ann, ((pad, pad), (pad, pad)), 'symmetric')

        # set pixel values to label
        ann = ann - 1
        ann[ ann < 0 ] = 0
        print ann.shape
        print img.shape

        print 'min-max'
        print np.min( ann ), np.max( ann )
        print np.min( img ), np.max( img )


        #nsamples_perImage = int(n_samples_remaining/(n_images - img_index))
        #n_img_samples = min( len(indices[0]), nsamples_perImage)
        n_samples_per_image = int(n_samples_remaining/(n_images - i_image))
        #n_n_img_samples = nsamples_perImage

        

        print '--------'
        print 'i_image:',i_image
        print 'image',files_gray[i_image]
        #print 'n_samples_remaining:', n_samples_remaining
        #print 'n_samples_per_image:', n_samples_per_image
        #print 'n_samples_per_image_label:',n_samples_per_image_label
        print n_samples_per_label


        # sample equally from each label
        for i_label in range( n_labels ):

            n_samples = int(n_samples_per_label[ i_label ]/(n_images - i_image))

            if n_samples == 0:
                continue
            
            coordinates = annotations[ i_label ]

            # stay within bounds of available labels
            n_samples = min( len(coordinates)/2, n_samples )
            print '--'
            print 'i_label:', i_label
            print 'n_samples:', n_samples

            i_coord = 0
            for i_sample in range( n_samples ):
                (row, col) = (coordinates[i_coord], coordinates[i_coord+1])
                print i_label, row, col

                imgPatch = img[row:row+patchSize, col:col+patchSize]
                annPatch = ann[row:row+patchSize, col:col+patchSize]

                print 'sample#:', counter
                print 'original'
                print np.unique(annPatch)
                print annPatch.flatten()
                if random.random() < 0.5:
                    imgPatch = np.fliplr(imgPatch)
                    annPatch = np.fliplr(annPatch)

                rotateInt = random.randint(0,3)
                imgPatch = np.rot90(imgPatch, rotateInt)
                annPatch = np.rot90(annPatch, rotateInt)

                imgPatch, annPatch = deform_images( imgPatch, annPatch )
                annPatch = np.round( annPatch )

                imgPatch = imgPatch / np.double(np.max(imgPatch))
                #annPatch = annPatch / np.double(np.max(annPatch))            

                offset_small_patch = int(np.ceil((patchSize - outPatchSize) / 2.0))
                annPatch = annPatch[offset_small_patch:offset_small_patch+outPatchSize,
                            offset_small_patch:offset_small_patch+outPatchSize]

                whole_set_patches[counter,:] = imgPatch.flatten()
                whole_set_labels[counter] = np.int32(annPatch.flatten())
                #whole_set_labels[counter] = np.int32(annPatch.flatten() > 0)

                print 'modified:'
                print 'row:', row, 'col:', col
                print 'patch'
                print whole_set_patches[counter,:]
                print np.min( whole_set_patches[counter,:] ), np.max( whole_set_patches[counter,:] )
                print 'labels'
                print whole_set_labels[counter]
                print np.unique( whole_set_labels[counter] )
                counter += 1


                # advance to next coordinate
                i_coord += 2

            #n_img_samples -= n_label_samples
            #n_samples_remaining -= n_label_samples
            n_samples_per_label[ i_label ] -= n_samples
        


        if True:
            continue

    
        for i in range(n_img_samples):

            randmem = random.choice(xrange(len(indices[0])))
            (row,col) = (indices[0][randmem], indices[1][randmem])
            print 'sampling from...', row, col
            print 'img:', files_gray[img_index]

            imgPatch = img[row:row+patchSize, col:col+patchSize]
            annPatch = ann[row:row+patchSize, col:col+patchSize]

            print 'original'
            print np.unique(annPatch)
            print annPatch.flatten()
            if random.random() < 0.5:
                imgPatch = np.fliplr(imgPatch)
                annPatch = np.fliplr(annPatch)

            rotateInt = random.randint(0,3)
            imgPatch = np.rot90(imgPatch, rotateInt)
            annPatch = np.rot90(annPatch, rotateInt)

            imgPatch, annPatch = deform_images( imgPatch, annPatch )
            annPatch = np.round( annPatch )

            imgPatch = imgPatch / np.double(np.max(imgPatch))
            #annPatch = annPatch / np.double(np.max(annPatch))            

            offset_small_patch = int(np.ceil((patchSize - outPatchSize) / 2.0))
            annPatch = annPatch[offset_small_patch:offset_small_patch+outPatchSize, 
                        offset_small_patch:offset_small_patch+outPatchSize]

            whole_set_patches[counter,:] = imgPatch.flatten()
            whole_set_labels[counter] = np.int32(annPatch.flatten())
            #whole_set_labels[counter] = np.int32(annPatch.flatten() > 0)

            print 'modified:'
            print 'row:', row, 'col:', col
            print 'patch'
            print whole_set_patches[counter,:]
            print np.min( whole_set_patches[counter,:] ), np.max( whole_set_patches[counter,:] )
            print 'labels'
            print whole_set_membranes[counter]
            print np.unique( whole_set_membranes[counter] )

            counter += 1

        print counter
        print '-----'
        n_samples_remaining -= n_img_samples

    print 'counter:', counter
    print 'n_samples_per_label:', n_samples_per_label

    #normalize data
    whole_data = np.float32(whole_set_patches)
    whole_data = whole_data - 0.5

    print np.max(whole_data), np.min(whole_data)
    print np.max(whole_set_labels), np.min(whole_set_labels)

    data = whole_data.copy()
    labels = whole_set_labels.copy()

    #remove the sorting in image order
    shuffleIndex = np.random.permutation(np.shape(labels)[0])
    for i in xrange(np.shape(labels)[0]):
        whole_data[i,:] = data[shuffleIndex[i],:]
        whole_set_labels[i,:] = labels[shuffleIndex[i],:]
        
    data_set = (whole_data, whole_set_labels)

    end_time = time.time()
    total_time = (end_time - start_time)
    print 'Running time: ', total_time / 60.
    print 'finished sampling data'

    return data_set


# changed the patch sampling to use upper left corner instead of middle pixel
# for patch labels it doesn't matter and it makes sampling even and odd patches easier
def generate_experiment_data_patch_prediction(purpose='train', nsamples=1000, patchSize=29, outPatchSize=1):
    print 'generate_experiment_data_patch_prediction'
    exit(1)

    def relabel(image):
        id_list = np.unique(image)
        for index, id in enumerate(id_list):
            image[image==id] = index
        return image

    start_time = time.time()

    pathPrefix = '/media/vkaynig/Data2/Cmor_paper_data/not_normalized/'
#    pathPrefix = '/media/vkaynig/Data1/Cmor_paper_data/Thalamus-LGN/Data/25-175_train/'
#pathPrefix = '/media/vkaynig/Data1/Cmor_paper_data/Cerebellum-P7/Dense/'
#    pathPrefix = '/media/vkaynig/Data1/Cmor_paper_data/Cortex-ECS/'
    if not os.path.exists(pathPrefix):
        pathPrefix = '/n/pfister_lab/vkaynig/'

    # if purpose=='train':
    #     if np.random.random()>0.5:
    #         pathPrefix = pathPrefix + 'dense40_train/'
    #     else:
    #         pathPrefix = pathPrefix + 'dense49_train/'
    # else:
    #     pathPrefix = pathPrefix + 'dense40_train/'

    print "#################################"
    print purpose
    print pathPrefix

    img_search_string_membraneImages = pathPrefix + 'labels/membranes_fullContour/' + purpose + '/*.tif'
    img_search_string_labelImages = pathPrefix + 'labels/' + purpose + '/*.tif'
    img_search_string_grayImages = pathPrefix + 'images/' + purpose + '/*.tif'


    #<felix-addition>
    pathPrefix = '/n/home00/fgonda/icon/data/reference/'

    #img_search_string_membraneImages = pathPrefix + 'labels/membranes_fullContour/' + purpose + '/*.tif'
    img_search_string_membraneImages = pathPrefix + 'labels/membranes/' + purpose + '/*.tif'
    img_search_string_labelImages = pathPrefix + 'labels/' + purpose + '/*.tif'
    img_search_string_grayImages = pathPrefix + 'images/' + purpose + '/*.tif'
    #</felix-addition>
    img_files_gray = sorted( glob.glob( img_search_string_grayImages ) )
    img_files_membrane = sorted( glob.glob( img_search_string_membraneImages ) )
    img_files_labels = sorted( glob.glob( img_search_string_labelImages ) )

    print len(img_files_gray)
    print len(img_files_membrane)
    print len(img_files_labels)

    whole_set_patches = np.zeros((nsamples, patchSize**2), dtype=np.float)
    whole_set_labels = np.zeros((nsamples, outPatchSize**2), dtype=np.int32)
    whole_set_membranes = np.zeros((nsamples, outPatchSize**2), dtype=np.int32)

    #how many samples per image?
    nsamples_perImage = np.uint(np.ceil( 
            (nsamples) / np.float(np.shape(img_files_gray)[0])
            )) 
    print 'using ' + np.str(nsamples_perImage) + ' samples per image.'

    img = mahotas.imread(img_files_gray[0])
    grayImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
    labelImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
    membraneImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
    maskImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))

    # read the data
    # in random order
    read_order = np.random.permutation(np.shape(img_files_gray)[0])
    read_order = read_order[:nsamples]
    for counter, img_index in enumerate(read_order):
        #print img_files_gray[img_index]
        img = mahotas.imread(img_files_gray[img_index])
        # normalizes [0,1]
        img = normalizeImage(img, doClahe=True) 
        grayImages[:,:,counter] = img
        membrane_img = mahotas.imread(img_files_membrane[img_index])/255.        
        membraneImages[:,:,counter] = membrane_img
        maskImages[:,:,counter] = 1.0
        if purpose == 'validate':
            label_img = mahotas.imread(img_files_labels[img_index])        
            label_img = np.double(label_img)
            if label_img.ndim == 3:
                label_img = label_img[:,:,0] + 256*label_img[:,:,1] + 256**2 * label_img[:,:,2]
            labelImages[:,:,counter] = label_img
        

    print
    counter = 0
    for img_index in xrange(nsamples):#np.shape(img_files_gray)[0]):
        #print img_files_gray[read_order[img_index]]
        img = grayImages[:,:,img_index]        
        label_img = labelImages[:,:,img_index]
        membrane_img = membraneImages[:,:,img_index]
        mask_img = maskImages[:,:,img_index]

        if purpose=='train':
           membrane_img = adjust_imprecise_boundaries(img, membrane_img, 0)

        #get rid of invalid image borders
        mask_img[:,-(patchSize-1):] = 0
        mask_img[-(patchSize-1):,:] = 0

        valid_indices = np.nonzero(mask_img)

        for i in xrange(nsamples_perImage):
            if counter >= nsamples:
                break

            randmem = random.choice(xrange(len(valid_indices[0])))
            (row,col) = (valid_indices[0][randmem], 
                         valid_indices[1][randmem])

            imgPatch = img[row:row+patchSize, col:col+patchSize]
            membranePatch = membrane_img[row:row+patchSize, col:col+patchSize]
            labelPatch = label_img[row:row+patchSize, col:col+patchSize]

            print 'sample#:', counter
            print 'original'
            print np.unique(imgPatch)
            print imgPatch.flatten()
            if random.random() < 0.5:
                imgPatch = np.fliplr(imgPatch)
                membranePatch = np.fliplr(membranePatch)
                if purpose == 'validate':
                    labelPatch = np.fliplr(labelPatch)

            rotateInt = random.randint(0,3)
            imgPatch = np.rot90(imgPatch, rotateInt)
            membranePatch = np.rot90(membranePatch, rotateInt)
            if purpose=='validate':
                labelPatch = np.rot90(labelPatch, rotateInt)

            if purpose=='validate':
                labelPatch = relabel(labelPatch)
                imgPatch, membranePatch = deform_images(imgPatch, membranePatch)
                # get rid of interpolation artifacts
                membranePatch = np.round(membranePatch)
                membranePatch, _ = mahotas.label(1-membranePatch)
            else:
                imgPatch, membranePatch = deform_images(imgPatch, membranePatch)
                # get rid of interpolation artifacts
                membranePatch = np.round(membranePatch)


            imgPatch = imgPatch / np.double(np.max(imgPatch))
            membranePatch = membranePatch / np.double(np.max(membranePatch))

            # crop labelPatch to potentially smaller output size
            offset_small_patch = int(np.ceil((patchSize - outPatchSize) / 2.0))
            membranePatch = membranePatch[offset_small_patch:offset_small_patch+outPatchSize, 
                                    offset_small_patch:offset_small_patch+outPatchSize]
            labelPatch = labelPatch[offset_small_patch:offset_small_patch+outPatchSize, 
                                    offset_small_patch:offset_small_patch+outPatchSize]

            whole_set_patches[counter,:] = imgPatch.flatten()
            whole_set_labels[counter] = labelPatch.flatten()
            whole_set_membranes[counter] = np.int32(membranePatch.flatten() > 0)

            print 'row:', row, 'col:', col
            print 'patch'
            print whole_set_patches[counter,:]
            print np.min( whole_set_patches[counter,:] ), np.max( whole_set_patches[counter,:] )
            print 'labels'
            print whole_set_membranes[counter]
            print np.unique( whole_set_membranes[counter] )
            #print np.unique(whole_set_patches[counter,:])
            counter += 1

    #normalize data
    whole_data = np.float32(whole_set_patches)
    whole_data = whole_data - 0.5

    print np.max(whole_data), np.min(whole_data)
    print np.max(whole_set_membranes), np.min(whole_set_membranes)

    data = whole_data.copy()
    labels = whole_set_labels.copy()
    membranes = whole_set_membranes.copy()

    #remove the sorting in image order
    shuffleIndex = np.random.permutation(np.shape(membranes)[0])
    for i in xrange(np.shape(membranes)[0]):  
        whole_data[i,:] = data[shuffleIndex[i],:]
        whole_set_labels[i,:] = labels[shuffleIndex[i],:]
        whole_set_membranes[i,:] = membranes[shuffleIndex[i],:]
    
    if purpose == 'validate':
        data_set = (whole_data, whole_set_membranes, whole_set_labels)    
    else:
        data_set = (whole_data, whole_set_membranes)    

    end_time = time.time()
    total_time = (end_time - start_time)
    print 'Running time: ', total_time / 60.
    print 'finished sampling data'

    return data_set



def gen_validation_data(project, nsamples=1000, patchSize=29, outPatchSize=1):
    def relabel(image):
        id_list = np.unique(image)
        for index, id in enumerate(id_list):
            image[image==id] = index
        return image

    return None

def generate_experiment_data_patch_prediction_layers(purpose='train', nsamples=1000, patchSize=29, outPatchSize=1, nr_layers=3):
    def relabel(image):
        id_list = np.unique(image)
        for index, id in enumerate(id_list):
            image[image==id] = index
        return image

    start_time = time.time()

    if os.path.exists('/media/vkaynig/Data1/Cmor_paper_data/'):
        pathPrefix = '/media/vkaynig/Data1/Cmor_paper_data/'
    else:
        pathPrefix = '/n/pfister_lab/vkaynig/'

    img_search_string_membraneImages = pathPrefix + 'labels/membranes_fullContour/' + purpose + '/*.tif'
    img_search_string_labelImages = pathPrefix + 'labels/' + purpose + '/*.tif'
    img_search_string_grayImages = pathPrefix + 'images/' + purpose + '/*.tif'

    img_files_gray = sorted( glob.glob( img_search_string_grayImages ) )
    img_files_membrane = sorted( glob.glob( img_search_string_membraneImages ) )
    img_files_labels = sorted( glob.glob( img_search_string_labelImages ) )

    whole_set_patches = np.zeros((nsamples, nr_layers, patchSize**2), dtype=np.float)
    whole_set_labels = np.zeros((nsamples, outPatchSize**2), dtype=np.int32)
    whole_set_membranes = np.zeros((nsamples, outPatchSize**2), dtype=np.int32)

    #how many samples per image?
    nsamples_perImage = np.uint(np.ceil( 
            (nsamples) / np.float(np.shape(img_files_gray)[0])
            )) 
    print 'using ' + np.str(nsamples_perImage) + ' samples per image.'
    counter = 0

    img = mahotas.imread(img_files_gray[0])
    grayImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
    labelImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
    membraneImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))
    maskImages = np.zeros((img.shape[0],img.shape[1], np.shape(img_files_gray)[0]))

    # read the data
    # in random order
    #read_order = np.random.permutation(np.shape(img_files_gray)[0])
    for img_index in range(np.shape(img_files_gray)[0]):
        #print img_files_gray[img_index]
        img = mahotas.imread(img_files_gray[img_index])
        # normalizes [0,1]
        img = normalizeImage(img) 
        grayImages[:,:,img_index] = img
        membrane_img = mahotas.imread(img_files_membrane[img_index])/255.        
        membraneImages[:,:,img_index] = membrane_img
        maskImages[:,:,img_index] = 1.0
        if purpose == 'validate':
            label_img = mahotas.imread(img_files_labels[img_index])        
            label_img = np.double(label_img)
            labelImages[:,:,img_index] = label_img
            
    for img_index in xrange(np.shape(img_files_gray)[0]):
        img_cs = int(np.floor(nr_layers/2))
        img_valid_range_indices = np.clip(range(img_index-img_cs,img_index+img_cs+1),0,np.shape(img_files_gray)[0]-1)
        img = grayImages[:,:,img_valid_range_indices]     
        label_img = labelImages[:,:,img_index]
        membrane_img = membraneImages[:,:,img_index]
        mask_img = maskImages[:,:,img_index]

        if purpose=='train':
            # adjust according to middle image
            membrane_img = adjust_imprecise_boundaries(img[:,:,img_cs], membrane_img, 0)

        #get rid of invalid image borders
        mask_img[:,-patchSize:] = 0
        mask_img[-patchSize:,:] = 0

        valid_indices = np.nonzero(mask_img)

        for i in xrange(nsamples_perImage):
            
            if counter >= nsamples:
                break

            randmem = random.choice(xrange(len(valid_indices[0])))
            (row,col) = (valid_indices[0][randmem], 
                         valid_indices[1][randmem])

            imgPatch = img[row:row+patchSize, col:col+patchSize,:]
            membranePatch = membrane_img[row:row+patchSize, col:col+patchSize]
            labelPatch = label_img[row:row+patchSize, col:col+patchSize]

            if random.random() < 0.5:
                for flip_i in xrange(nr_layers):
                    imgPatch[:,:,flip_i] = np.fliplr(imgPatch[:,:,flip_i])
                membranePatch = np.fliplr(membranePatch)
                if purpose == 'validate':
                    labelPatch = np.fliplr(labelPatch)

            rotateInt = random.randint(0,3)
            for rot_i in xrange(nr_layers):
                imgPatch[:,:,rot_i] = np.rot90(imgPatch[:,:,rot_i], rotateInt)
            membranePatch = np.rot90(membranePatch, rotateInt)
            if purpose=='validate':
                labelPatch = np.rot90(labelPatch, rotateInt)

            if purpose=='validate':
                labelPatch = relabel(labelPatch)
                deformed_images = deform_images_list(np.dstack([imgPatch*255, np.reshape(membranePatch*255,(patchSize,patchSize,1)), np.uint8(np.reshape(labelPatch,(patchSize,patchSize,1)))]))
                imgPatch, membranePatch, labelPatch = np.split(deformed_images,[imgPatch.shape[2],imgPatch.shape[2]+1], axis=2)
            else:
                deformed_images = deform_images_list(np.dstack([imgPatch*255, np.reshape(membranePatch,(patchSize,patchSize,1))*255]))
                imgPatch, membranePatch = np.split(deformed_images,[imgPatch.shape[2]], axis=2)            

            imgPatch = imgPatch / np.double(np.max(imgPatch))
            membranePatch = membranePatch / np.double(np.max(membranePatch))

            # crop labelPatch to potentially smaller output size
            offset_small_patch = int(np.ceil((patchSize - outPatchSize) / 2.0))
            membranePatch = membranePatch[offset_small_patch:offset_small_patch+outPatchSize, 
                                    offset_small_patch:offset_small_patch+outPatchSize]
            labelPatch = labelPatch[offset_small_patch:offset_small_patch+outPatchSize, 
                                    offset_small_patch:offset_small_patch+outPatchSize]

            #whole_set_patches = np.zeros((nsamples, nr_layers, patchSize**2), dtype=np.float)
            for patch_i in xrange(nr_layers):
                whole_set_patches[counter,patch_i,:] = imgPatch[:,:,patch_i].flatten()
            whole_set_labels[counter] = labelPatch.flatten()
            whole_set_membranes[counter] = np.int32(membranePatch.flatten() > 0)
            counter += 1

    #normalize data
    whole_data = np.float32(whole_set_patches)
    whole_data = whole_data - 0.5

    data = whole_data.copy()
    labels = whole_set_labels.copy()
    membranes = whole_set_membranes.copy()

    #remove the sorting in image order
    shuffleIndex = np.random.permutation(np.shape(membranes)[0])
    for i in xrange(np.shape(membranes)[0]):  
        whole_data[i,:,:] = data[shuffleIndex[i],:,:]
        whole_set_labels[i,:] = labels[shuffleIndex[i],:]
        whole_set_membranes[i,:] = membranes[shuffleIndex[i],:]
    
    if purpose == 'validate':
        data_set = (whole_data, whole_set_membranes, whole_set_labels)    
    else:
        data_set = (whole_data, whole_set_membranes)    

    end_time = time.time()
    total_time = (end_time - start_time)
    print 'Running time: ', total_time / 60.
    print 'finished sampling data'

    return data_set

if __name__=="__main__":
    import uuid

    test = generate_experiment_data_patch_prediction(purpose='validate', nsamples=20, patchSize=1024, outPatchSize=1024)

    # dir_path = './training_patches/'
    
    # for i in xrange(30):
    #     unique_filename = str(uuid.uuid4())
    #     img = np.reshape(test[1][i],(388,388))
    #     img_gray = np.reshape(test[0][i],(572,572))
    #     mahotas.imsave(dir_path+unique_filename+'.tif', np.uint8(img*255))
    #     mahotas.imsave(dir_path+unique_filename+'_gray.tif', np.uint8((img_gray+0.5)*255))
        

    #data_val = generate_experiment_data_supervised(purpose='validate', nsamples=10000, patchSize=65, balanceRate=0.5)
    #data = generate_experiment_data_patch_prediction(purpose='validate', nsamples=2, patchSize=315, outPatchSize=215)
    # plt.imshow(np.reshape(data[0][0],(315,315))); plt.figure()
    # plt.imshow(np.reshape(data[1][0],(215,215))); plt.figure()
    # plt.imshow(np.reshape(data[2][0],(215,215))); plt.show()

    # image = mahotas.imread('ac3_input_0141.tif')
    # image = normalizeImage(image)
    # label = mahotas.imread('ac3_labels_0141.tif') / 255.
    # test = adjust_imprecise_boundaries(image, label, 10)

    # plt.imshow(label+image); plt.show()
    # plt.imshow(test+image); plt.show()
